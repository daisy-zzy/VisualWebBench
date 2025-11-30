import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

from model_adapters import BaseAdapter
from utils.constants import *
from utils.helper_tools import *
from utils.visual_CoT_agent import VisualCoTAgent


class DeepseekAgentAdapter(BaseAdapter):
    """
    DeepSeek-VL adapter with optional VisualCoTAgent for WebQA.

    - For WEBQA_TASK and use_agent=True:
        Uses VisualCoTAgent (multi-view cropping) with the raw question,
        then refines the chosen final answer to a minimal phrase.
    - For HEADING_OCR_TASK (with use_agent=True):
        Uses full-page heading summary + top-strip crop + a model-based
        decision over the final heading.
    - For all other tasks:
        Simple one-shot DeepSeek call with (prompt, image).
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 48,
        use_agent: bool = False,
        agent: Optional[VisualCoTAgent] = None,
        temperature: float = 0.0,
        agent_grid_size: int = 3,
        agent_max_crops: int = 1,
        agent_margin_frac_of_cell: float = 0.2,
        agent_save_dir: Optional[str] = None,
    ):
        super().__init__(model, tokenizer)
        self.processor = VLChatProcessor.from_pretrained(tokenizer.name_or_path)
        self.max_new_tokens = max_new_tokens
        self.model.eval()

        # Agent setup
        self.use_agent = use_agent
        if use_agent:
            if agent is not None:
                self.agent = agent
            else:
                self.agent = VisualCoTAgent(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    max_new_tokens=max_new_tokens,
                    grid_size=agent_grid_size,
                    max_crops=agent_max_crops,
                    margin_frac_of_cell=agent_margin_frac_of_cell,
                    save_dir=agent_save_dir,
                )
        else:
            self.agent = None

    # ---- core one-shot DeepSeek call ----
    def _deepseek_generate(
        self,
        query: str,
        image: Image,
    ) -> str:
        """Single multimodal generation using DeepSeek (no cropping)."""
        image = image.convert("RGB")

        # Prepare inputs
        conv = [
            {
                "role": "User",
                "content": "<image_placeholder>" + query,
                "images": [image],
            },
            {"role": "Assistant", "content": ""},
        ]
        prepared = self.processor(
            conversations=conv,
            images=[image],
            force_batchify=True,
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepared)
        out_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared.attention_mask,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )

        decoded = self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        outputs = decoded.strip()

        return outputs

    # ================== Action grounding helpers (letter-labeled boxes) ================== #

    def _scan_region_for_labels(
        self,
        region_image: Image.Image,
        region_tag: str,
        instruction: str,
    ) -> str:
        """
        Run on a cropped region (e.g. TL / TR / BL / BR).

        Asks the model to:
          - find red, letter-labeled boxes,
          - read / summarize the content INSIDE each red box.
        """
        prompt = (
            "You are looking at ONE PART of a web page screenshot.\n"
            "In this dataset, CANDIDATE ACTION REGIONS are drawn as rectangles with a BRIGHT RED BORDER.\n"
            "Each of these red rectangles has a SINGLE WHITE CAPITAL LETTER label (A, B, C, ...)\n"
            "placed near one of its corners, usually with a dark or black highlight.\n\n"
            "Your tasks in THIS REGION ONLY:\n"
            "1. Find all red, letter-labeled rectangles that are at least partly visible.\n"
            "2. For each letter label, describe what is INSIDE that red box and what its main purpose is\n"
            "  \n\n"
            "Important rules:\n"
            "- Only treat a letter as a label if it is a SINGLE capital letter (A–Z) directly associated\n"
            "  with a red rectangular border.\n"
            "- Do NOT invent labels that are not clearly visible in this region.\n"
            "- Ignore letters that are part of normal text, words, paragraphs, or logos.\n"
            "- When you describe a label, focus on the UI element INSIDE the red box,\n"
            "  not the surrounding page.\n"
            "- If a red box is partly cut off by the crop, still include it if you can see enough\n"
            "  to guess its role.\n"
            "- Never write descriptions like 'NONE', 'no label', or 'no content' for a label.\n"
            "  If you truly see no red, letter-labeled rectangles at all, use the special output\n"
            "  described below.\n\n"
            "You will later be asked to choose ONE label that best satisfies this instruction:\n"
            f"INSTRUCTION: {instruction}\n\n"
            "Output format (MUST follow one of these exactly):\n"
            "1) If you see one or more valid red, letter-labeled rectangles in this region,\n"
            "   output ONE line per label, in any order:\n"
            "      LETTER: short description (<= 10 words)\n"
            "   where LETTER is a single capital letter A–Z with NO brackets or extra symbols.\n"
            "   For example:\n"
            "      A: a button that ....\n"
            "   Each label must appear at most once.\n\n"
            "2) If you do NOT see any red, letter-labeled rectangles in this region,\n"
            "   output exactly one line with:\n"
            "      NONE\n\n"
            "Do NOT add any other text, explanations, or JSON.\n"
            f"Current region tag: {region_tag}\n"
        )



        return self._deepseek_generate(prompt, region_image)


    def _parse_region_labels(
        self,
        raw_text: str,
        region_tag: str,
    ) -> List[Dict[str, str]]:
        """
        Parse output of _scan_region_for_labels into a list of dicts:
          { 'label': 'A', 'description': '...', 'region': 'TL' }
        """
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return []

        # If the model explicitly says NONE anywhere → treat as no labels
        joined = " ".join(lines).lower()
        if "none" in joined and all(len(ln) <= 8 for ln in lines):
            return []

        results: List[Dict[str, str]] = []
        for ln in lines:
            # Expect something like "A: description"
            if ":" not in ln:
                continue
            left, right = ln.split(":", 1)
            left = left.strip()
            right = right.strip()
            if not left or len(left) != 1 or not left.isalpha():
                continue
            label = left.upper()
            desc = right
            if not desc:
                continue
            results.append(
                {
                    "label": label,
                    "description": desc,
                    "region": region_tag,
                }
            )
        return results

    def _collect_action_candidates(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, str]:
        """
        Divide the page into a 2x2 grid (TL, TR, BL, BR), scan each region for labels,
        and aggregate descriptions per label.

        Returns:
            candidates: dict[label] = merged_description
        """
        # 2x2 quadrants in normalized coords
        regions = {
            "TL": [0.0, 0.0, 0.5, 0.5],
            "TR": [0.5, 0.0, 1.0, 0.5],
            "BL": [0.0, 0.5, 0.5, 1.0],
            "BR": [0.5, 0.5, 1.0, 1.0],
        }

        candidates: Dict[str, str] = {}

        for tag, bbox in regions.items():
            region_img = crop_normalized(image, bbox)
            print(f"[ActionGround][scan] Region={tag}, bbox={bbox}")
            raw = self._scan_region_for_labels(region_img, tag, instruction)
            print(f"[ActionGround][scan] raw output ({tag}):")
            print(raw)

            parsed = self._parse_region_labels(raw, tag)
            print(f"[ActionGround][scan] parsed labels ({tag}): {parsed}")

            for item in parsed:
                lbl = item["label"]
                desc = item["description"]
                # If the label appears multiple times (e.g. overlapping crops),
                # keep the longer / more informative description.
                if lbl not in candidates or len(desc) > len(candidates[lbl]):
                    candidates[lbl] = desc

        print("[ActionGround] aggregated candidates:")
        for lbl, desc in candidates.items():
            print(f"  {lbl}: {desc!r}")
        return candidates

    def _select_action_label(
        self,
        image: Image.Image,
        instruction: str,
        candidates: Dict[str, str],
    ) -> str:
        """
        Given a dict[label] = description, ask the model to pick exactly one label.
        The call sees the FULL PAGE screenshot again, plus the candidate summaries
        and the natural-language instruction.
        """
        if not candidates:
            print("[ActionGround] No candidates found; falling back to 'A'.")
            return "A"  # trivial fallback

        # Build candidate list string
        lines = [f"{lbl}: {desc}" for lbl, desc in sorted(candidates.items())]
        labels_str = "\n".join(lines)
        label_set_str = ", ".join(sorted(candidates.keys()))

        prompt = (
            "You are looking at the FULL screenshot of a web page.\n"
            "Several UI regions on this page are highlighted by rectangles with a BRIGHT RED BORDER.\n"
            "Each such red box has a SINGLE CAPITAL LETTER label (A, B, C, ...).\n\n"
            "From previous steps, we extracted a short description of what is inside each\n"
            "red, letter-labeled box:\n"
            f"{labels_str}\n\n"
            "User instruction:\n"
            f"{instruction}\n\n"
            "Your job:\n"
            "- Using BOTH the screenshot and the descriptions above, decide which ONE\n"
            "  labeled red box the user should select in order to follow the instruction.\n"
            "- Think about what the user wants to do (e.g., open an article, use a search box,\n"
            "  open a menu, view an animal card, etc.) and match it to the most appropriate region.\n"
            "- You MUST choose exactly ONE label from the provided set.\n\n"
            "Output format (MUST follow exactly):\n"
            "  ANSWER: <LABEL> | reason: <very short reason>\n\n"
            f"where <LABEL> MUST be one of: {label_set_str}.\n"
            "Do NOT output any other text, JSON, or multiple labels.\n"
        )

        raw = self._deepseek_generate(prompt, image)
        print("[ActionGround][select] raw output:")
        print(raw)

        # Parse the last non-empty line
        lines_out = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines_out:
            chosen = sorted(candidates.keys())[0]
            print(f"[ActionGround][select] Empty output; fallback to {chosen}")
            return chosen

        last = lines_out[-1]
        if "ANSWER:" in last:
            _, rest = last.split("ANSWER:", 1)
            last = rest.strip()

        # Strip everything after first '|' if present
        if "|" in last:
            last = last.split("|", 1)[0].strip()

        # We expect the remaining piece to contain a single label
        label_char = ""
        for ch in last:
            if ch.isalpha():
                label_char = ch.upper()
                break

        if not label_char:
            chosen = sorted(candidates.keys())[0]
            print(f"[ActionGround][select] No valid label parsed; fallback to {chosen}")
            return chosen

        if label_char not in candidates:
            chosen = sorted(candidates.keys())[0]
            print(
                f"[ActionGround][select] Parsed label {label_char!r} not in candidates; "
                f"fallback to {chosen}"
            )
            return chosen

        print(f"[ActionGround][select] Final chosen label: {label_char}")
        return label_char


    def _run_action_grounding(
        self,
        image: Image.Image,
        instruction: str,
    ) -> str:
        """
        Full action grounding pipeline:

          1) Split page into 2x2 regions (TL, TR, BL, BR).
          2) For each region, detect letter labels + descriptions.
          3) Aggregate candidates by label.
          4) Ask the model to pick exactly ONE label.

        Returns:
            A single uppercase letter (A–Z) as the predicted action region label.
        """
        print("\n[ActionGround] ====== START ACTION GROUNDING ======")
        candidates = self._collect_action_candidates(image, instruction)

        if not candidates:
            # As a very last resort, use a one-shot guess based on the full page.
            # But still clamp to 'A' for safety (dataset may expect a letter).
            print("[ActionGround] No candidates after scanning; return 'A' as trivial guess.")
            return "A"

        chosen_label = self._select_action_label(image, instruction, candidates)
        print(f"[ActionGround] ====== FINAL ACTION LABEL: {chosen_label} ======")
        return chosen_label


    # ---- WebQA answer refiner ----
    def _refine_answer(
        self,
        question: str,
        raw_answer: str,
        image: Image.Image,
    ) -> str:
        """
        Use DeepSeek once more to trim the chosen answer
        to the shortest phrase that directly answers the question.
        """
        if not raw_answer or not raw_answer.strip():
            return ""

        prompt = (
            "You are given a QUESTION and a CANDIDATE ANSWER that were extracted from a web page.\n"
            "Your job is to trim the candidate answer to the SHORTEST phrase that still correctly\n"
            "answers the question.\n\n"
            f"QUESTION: {question}\n"
            f"CANDIDATE ANSWER: {raw_answer}\n\n"
            "Return ONLY the trimmed answer phrase.\n"
            "Do NOT add any explanation or extra words.\n"
        )

        text = self._deepseek_generate(prompt, image)

        # Light cleaning: remove surrounding quotes/whitespace.
        trimmed = text.strip().strip('"').strip("'").strip()
        if not trimmed:
            return raw_answer
        return trimmed

    # ---- Heading OCR summary call ----
    def _heading_summary_call(self, image: Image.Image) -> str:
        """
        Ask the model to list main headings on the page (full screenshot).
        We want exact literal texts, not paraphrases.
        """
        prompt = (
            "You are looking at a webpage screenshot.\n"
            "List up to 3 main heading texts you see, from the most prominent to less prominent.\n\n"
            "Important:\n"
            "- Copy the heading texts exactly as they appear on the page.\n"
            "- Do NOT paraphrase.\n"
            "- Do NOT add any explanation.\n\n"
            "Format:\n"
            "1. <heading 1>\n"
            "2. <heading 2>\n"
            "3. <heading 3>\n"
        )

        return self._deepseek_generate(prompt, image)
    

    # ---- Heading OCR: model-based final decision ----
    def _decide_heading(
        self,
        image: Image.Image,
        summary_heading: str,
        crop_heading: str,
    ) -> str:
        """
        Given candidate headings from:
          - full-page summary (summary_heading)
          - top-strip crop (crop_heading),

        ask the model to decide which one is the true main content heading,
        or output a different heading read directly from the page if both
        candidates are wrong.

        This is where we disambiguate main content heading vs. grand site title.
        """
        if not (summary_heading or crop_heading):
            return ""

        a = summary_heading or ""
        b = crop_heading or ""

        prompt = (
            "You are looking at a screenshot of a web page.\n"
            "Your task is to find the MAIN CONTENT HEADING of the page.\n\n"
            "Important:\n"
            "- The main content heading is the title of the main article or main content section.\n"
            "- Do NOT pick the site name, logo text, company name, browser tab title,\n"
            "  or generic banner slogan.\n"
            "- Focus on the heading that appears above the main body of content.\n\n"
            f"CANDIDATE_A: {a if a else '(none)'}\n"
            f"CANDIDATE_B: {b if b else '(none)'}\n\n"
            "Steps you should follow (do NOT write these steps out):\n"
            "1. Look carefully at the screenshot and locate the true main content heading.\n"
            "2. If CANDIDATE_A or CANDIDATE_B exactly match that heading, choose that candidate.\n"
            "3. If neither candidate is correct, read the correct main heading from the page and output it.\n\n"
            "You MUST respond in exactly ONE line:\n"
            "FINAL_HEADING: <exact heading text copied from the page>\n"
            "Do NOT add any explanation or extra text.\n"
        )

        text = self._deepseek_generate(prompt, image).strip()
        if not text:
            return clean_heading(a or b)

        # Take last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        last = lines[-1] if lines else text

        if "FINAL_HEADING" in last.upper():
            # Be tolerant to 'FINAL_HEADING:' or 'Final_Heading :'
            parts = last.split(":", 1)
            val = parts[1].strip() if len(parts) > 1 else ""
        else:
            val = last.strip()

        return clean_heading(val)

    # ---- main logic used by __call__ ----
    def generate(
        self,
        query: str,
        image: Image.Image,
        task_type: str = "",
        question: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Core entry point used by BaseAdapter.__call__.

        For HEADING_OCR_TASK + use_agent=True:
            - Use heading summary (full page) + top-strip crop.
            - Then let the model choose the final heading via _decide_heading.
        For WEBQA_TASK + use_agent=True:
            - Use VisualCoTAgent and take its final_answer.
            - Then run a refinement pass to trim the answer.
        Otherwise:
            - Fall back to simple one-shot generation.
        """
        try:
            # ---------- Heading OCR pipeline ----------
            if (
                task_type == HEADING_OCR_TASK
                and self.use_agent
                and self.agent is not None
            ):
                print("\n[DeepseekAgentAdapter][HEADING] Using heading OCR pipeline (summary + top strip + model decision)")

                # (a) Full-page heading summary
                raw_summary = self._heading_summary_call(image)
                print("[DeepseekAgentAdapter][HEADING] raw_summary:")
                print(raw_summary)

                summary_heading = extract_top_heading_from_summary(raw_summary)
                print(f"[DeepseekAgentAdapter][HEADING] summary_heading: {summary_heading!r}")

                # (b) Top-strip crop (e.g. top 35% of the image)
                top_bbox = [0.0, 0.0, 1.0, 0.35]
                print(f"[DeepseekAgentAdapter][HEADING] top bbox: {top_bbox}")
                heading_view = crop_normalized(image, top_bbox)

                # use the provided heading_ocr_prompt as `query`
                raw_crop = self._deepseek_generate(query, heading_view)
                print("[DeepseekAgentAdapter][HEADING] raw_crop:")
                print(raw_crop)

                crop_heading = clean_heading(raw_crop)
                print(f"[DeepseekAgentAdapter][HEADING] crop_heading: {crop_heading!r}")

                if not (summary_heading or crop_heading):
                    print("[DeepseekAgentAdapter][HEADING] No candidates! Falling back to raw outputs.")
                    return crop_heading or summary_heading or raw_crop or raw_summary or ""

                # (c) Let the model decide which heading is correct
                final_heading = self._decide_heading(image, summary_heading, crop_heading)
                print(f"[DeepseekAgentAdapter][HEADING] final_heading: {final_heading!r}")
                return final_heading
            

            # ---------- WebQA multi-view pipeline + refiner ----------
            if (
                task_type == WEBQA_TASK
                and self.use_agent
                and self.agent is not None
            ):
                q = question if question is not None else query
                result = self.agent.run_chain(image, q)
                raw_final = result.get("final_answer", "") or ""
                print(f"[DeepseekAgentAdapter][WEBQA] raw_final: {raw_final!r}")
                
                return raw_final
                # refined = self._refine_answer(q, raw_final, image)
                # print(f"[DeepseekAgentAdapter][WEBQA] refined_final: {refined!r}")

                # return refined or raw_final
            
            if (
                task_type == ACTION_GROUND_TASK
                and self.use_agent
            ):
                q = question if question is not None else query
                return self._run_action_grounding(image, q)

            # ---------- Default: one-shot Deepseek for other tasks ----------
            return self._deepseek_generate(query, image)

        except Exception as e:
            print(f"[DeepseekAgentAdapter] Error during generation: {e}")
            return ""

    # ---- make adapter callable ----
    def __call__(
        self,
        query: str,
        image: Image.Image,
        task_type: str = "",
        **kwargs,
    ) -> str:
        """
        Allows using the adapter like:
            response = model_adapter(prompt, image, task_type=..., question=...)
        """
        return self.generate(query, image, task_type=task_type, **kwargs)
