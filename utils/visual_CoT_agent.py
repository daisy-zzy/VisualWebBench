import os
from typing import Optional, List, Dict, Any

import torch
from PIL import Image
from deepseek_vl.models import VLChatProcessor

from .helper_tools import *


# =================== VisualCoTAgent (original WebQA design) =================== #

class VisualCoTAgent:
    """
    Fixed-depth multi-view WebQA agent with k×k grid and margin cropping:

      - Views: full page + up to `max_crops` successive crops.
      - At each view:
          1) ANSWER-ONLY call that also outputs info_visible: YES/NO.
          2) If not last view: CROP policy (must pick exactly one grid cell).

      - Cropping:
          - Policy selects a cell (R<i>C<j>).
          - We expand that cell's bbox by a margin fraction of cell width/height,
            then crop + resize back to original image size.
          - This helps when the relevant text sits near the boundary between cells.

      - Final answer rule (last-YES-wins with graceful fallback):
          - If there exists at least one candidate with info_visible == YES,
              pick the **last** such candidate (later crops override earlier guesses).
          - If **no** candidate has info_visible == YES,
              fall back to the **first** candidate (the earliest global context guess).
    """

    def __init__(
        self,
        tokenizer,
        model,
        *,
        max_new_tokens: int = 48,
        save_dir: Optional[str] = None,
        grid_size: int = 3,
        max_crops: int = 1,
        margin_frac_of_cell: float = 0.2,
    ):
        """
        Args:
            tokenizer, model: pre-loaded deepseek tokenizer/model (we reuse the adapter's).
            max_new_tokens: max tokens per generation.
            save_dir: if not None, directory to save crops for debugging.
            grid_size: k for k×k grid (k >= 2).
            max_crops: maximum number of successive crops (views = 1 + max_crops).
            margin_frac_of_cell: how much to expand each chosen cell on each side,
                as a fraction of the cell size (e.g. 0.2 = expand by 20% of cell width/height).
        """
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")

        self.k = grid_size
        self.zone_to_bbox, self.synonyms = build_zone_mapping(self.k)

        self.max_crops = max_crops
        self.max_new_tokens = max_new_tokens
        self.margin_frac_of_cell = max(0.0, margin_frac_of_cell)

        self.processor = VLChatProcessor.from_pretrained(tokenizer.name_or_path)
        self.tokenizer = tokenizer
        self.model = model

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"[agent] Will save step images to: {self.save_dir}")

    # ------------------- Internal helpers ------------------- #

    def _zones_description(self) -> str:
        k = self.k
        desc = [
            f"The page is divided into a {k}x{k} grid of zones.",
            "Cells are named R<i>C<j>, where i is the row index (top=1, bottom=k),",
            "and j is the column index (left=1, right=k). For example, "
            f"R1C1 is top-left, R1C{k} is top-right, R{k}C1 is bottom-left, "
            f"and R{k}C{k} is bottom-right.",
        ]
        if k == 3:
            desc.append(
                "For k=3, you may also use the synonyms TL, TM, TR, ML, MM, MR, BL, BM, BR, "
                "where TL=R1C1, TM=R1C2, TR=R1C3, ML=R2C1, MM=R2C2, MR=R2C3, "
                "BL=R3C1, BM=R3C2, BR=R3C3."
            )
        return "\n".join(desc) + "\n"

    def _canonical_zone_name(self, zone: str) -> str:
        z = zone.upper()
        if z in self.zone_to_bbox:
            return z
        if z in self.synonyms:
            return self.synonyms[z]
        raise ValueError(f"Unknown zone name '{zone}' for grid_size={self.k}")

    def _expand_bbox_with_margin(self, bbox):
        """
        Expand a cell bbox by margin_frac_of_cell on each side,
        where the margin is expressed as a fraction of a single cell's
        width/height.
        """
        x0, y0, x1, y1 = bbox
        cell_w = 1.0 / self.k
        cell_h = 1.0 / self.k

        dx = self.margin_frac_of_cell * cell_w
        dy = self.margin_frac_of_cell * cell_h

        x0_exp = max(0.0, x0 - dx)
        y0_exp = max(0.0, y0 - dy)
        x1_exp = min(1.0, x1 + dx)
        y1_exp = min(1.0, y1 + dy)

        return [x0_exp, y0_exp, x1_exp, y1_exp]

    def _deepseek_call(
        self,
        image: Image.Image,
        instruction_text: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Single DeepSeek call → return raw decoded text.
        """
        image = image.convert("RGB")

        # Prepare inputs
        conv = [
            {
                "role": "User",
                "content": "<image_placeholder>" + instruction_text,
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
        with torch.inference_mode():
            out_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepared.attention_mask,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        decoded = self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        full_text = decoded.strip()

        return full_text

    # ------------------- ANSWER-ONLY prompt & parsing ------------------- #

    def _build_answer_instruction(
        self,
        question: str,
        view_tag: str,
        view_index: int,
        total_views: int,
    ) -> str:
        """
        Instruction for ANSWER-ONLY call:
        - General for any WebQA.
        - info_visible must be STRICT: YES only when the key info is clearly present and legible.
        """
        return (
            "You are a visual question answering model looking at a web page screenshot.\n"
            "Your task is to answer the user's question as accurately as possible based ONLY on this current view.\n\n"
            "You MUST respond in EXACTLY one line with the following format:\n"
            " ANSWER: <final answer> | info_visible: YES or NO | reason: <very short reason>\n\n"
            "The flag info_visible is VERY STRICT:\n"
            "- Set info_visible: YES ONLY if the key information needed to answer the question is clearly present,\n"
            "  fully readable, and directly visible in this view (for example, you can literally read the number,\n"
            "  name, or date that answers the question).\n"
            "- If you are guessing, only see partial context, or cannot clearly see the exact required value, you MUST\n"
            "  set info_visible: NO, even if you still attempt to give your best guess in the ANSWER.\n"
            "- When in doubt, choose info_visible: NO.\n\n"
            "Do not talk about cropping or future steps. Do not output JSON.\n\n"
            f"Question: {question}\n"
            f"Current view tag: {view_tag}\n"
            f"View index: {view_index} of {total_views-1}.\n"
        )

    def _split_answer_info_reason(self, line: str):
        """
        Parse line of form:
          ANSWER: <ans> | info_visible: YES/NO | reason: <reason>
        Be tolerant if the pattern is slightly off.
        """
        text = line.strip()

        # Strip leading 'ANSWER:'
        if "ANSWER:" in text:
            _, rest = text.split("ANSWER:", 1)
            text = rest.strip()

        # Defaults
        answer = text
        info_visible = "UNKNOWN"
        reason = ""

        parts = [p.strip() for p in text.split("|")]

        if parts:
            answer = parts[0].strip()

        for p in parts[1:]:
            low = p.lower()
            if "info_visible" in low:
                if ":" in p:
                    _, v = p.split(":", 1)
                    info_visible = v.strip().upper()
            elif "reason" in low:
                if ":" in p:
                    _, v = p.split(":", 1)
                    reason = v.strip()

        return answer, info_visible, reason

    def _parse_answer_candidate(self, raw_text: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return {
                "answer": "UNCERTAIN",
                "info_visible": "UNKNOWN",
                "reason": "Model returned empty output.",
                "raw_full": raw_text,
                "raw_line": "",
            }

        line = lines[-1]
        ans, info_visible, reason = self._split_answer_info_reason(line)
        return {
            "answer": ans,
            "info_visible": info_visible,
            "reason": reason,
            "raw_full": raw_text,
            "raw_line": line,
        }

    # ------------------- CROP policy prompt & parsing ------------------- #

    def _build_policy_instruction(
        self,
        question: str,
        view_tag: str,
        view_index: int,
        total_views: int,
    ) -> str:
        """
        Instruction for CROP policy:
        - It MUST choose exactly one CROP_ZONE.
        - No NO_CROP / STOP option.
        - Does NOT see previous answers.
        """
        zones_desc = self._zones_description()
        return (
            "You are a visual cropping policy for a web page screenshot.\n"
            "You see the current view (full page or a previous crop) and the question.\n"
            "Your ONLY job is to choose ONE grid cell to zoom into next.\n\n"
            + zones_desc +
            "You MUST respond in exactly ONE line with this format:\n"
            " CROP_ZONE R<i>C<j> | reason: <why this cell is the single best region to zoom>\n\n"
            "Guidelines:\n"
            "- Pretend this next crop is your FINAL zoom; choose the one region that is most likely to contain\n"
            "  missing or more detailed information needed to answer the question.\n"
            "- Focus on panels, text blocks, icons, or numbers that directly relate to the question.\n"
            "- Do NOT output NO_CROP or STOP. You must always choose a CROP_ZONE.\n\n"
            f"Question: {question}\n"
            f"Current view tag: {view_tag}\n"
            f"View index: {view_index} of {total_views-1}.\n"
        )

    def _parse_policy_line(self, raw_text: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return {
                "action": "ERROR",
                "reason": "Empty policy output.",
                "raw_full": raw_text,
                "raw_line": "",
            }

        line = lines[-1]

        if not line.startswith("CROP_ZONE"):
            return {
                "action": "ERROR",
                "reason": f"Policy did not start with CROP_ZONE: {line}",
                "raw_full": raw_text,
                "raw_line": line,
            }

        before, *after = line.split("|", 1)
        parts = before.split()
        if len(parts) < 2:
            return {
                "action": "ERROR",
                "reason": f"Missing zone in policy line: {line}",
                "raw_full": raw_text,
                "raw_line": line,
            }

        zone_raw = parts[1].strip()
        try:
            zone_canonical = self._canonical_zone_name(zone_raw)
            bbox = self.zone_to_bbox[zone_canonical]
        except Exception as e:
            return {
                "action": "ERROR",
                "reason": f"Invalid zone '{zone_raw}': {e}",
                "raw_full": raw_text,
                "raw_line": line,
            }

        reason = ""
        if after and "reason:" in after[0]:
            reason = after[0].split("reason:", 1)[1].strip()

        return {
            "action": "CROP",
            "zone": zone_canonical,
            "zone_raw": zone_raw,
            "bbox": bbox,
            "reason": reason,
            "raw_full": raw_text,
            "raw_line": line,
        }

    # ------------------- Main loop with last-YES-wins aggregation ------------------- #

    def run_chain(
        self,
        image: Image.Image,
        question: str,
        crop_dup_epsilon: float = 1e-3,
    ):
        """
        Main loop:

          - Number of views = 1 + max_crops:
              view 0: full image
              view 1: crop after policy_0
              view 2: crop after policy_1
              ...

          - At each view v:
              1) ANSWER-ONLY → candidate answer_v (with strict info_visible).
              2) If v < last_view: POLICY → CROP_ZONE → next view image
                 (bbox is expanded with margin before cropping).

          - Final answer:
              - If any candidate has info_visible in {YES, TRUE, VISIBLE}:
                    pick the *last* such candidate (later views override).
              - Else:
                    pick the *first* candidate (earliest global context guess).
        """
        original_img = image
        current_img = image

        history: List[Dict[str, Any]] = []
        answer_candidates: List[Dict[str, Any]] = []

        last_crop_bbox = None

        num_views = 1 + max(self.max_crops, 0)

        # Save the initial full image once for visualization
        if self.save_dir is not None:
            init_path = os.path.join(self.save_dir, "step_00_input.png")
            original_img.save(init_path)
            print(f"[agent] Saved initial full image to {init_path}")

        for view_idx in range(num_views):
            view_tag = "full" if view_idx == 0 else f"crop_{view_idx}"

            # ---------- 1) ANSWER-ONLY on current view ----------
            ans_instr = self._build_answer_instruction(
                question=question,
                view_tag=view_tag,
                view_index=view_idx,
                total_views=num_views,
            )
            raw_ans = self._deepseek_call(current_img, ans_instr)
            print("\n[agent] ---- FULL ASSISTANT OUTPUT (ANSWER-ONLY) ----")
            print(raw_ans)
            print("[agent] --------------------------------------------")

            cand = self._parse_answer_candidate(raw_ans)
            cand["step"] = view_idx
            cand["view"] = view_tag
            answer_candidates.append(cand)
            print(
                f"[agent] Answer-only candidate at view {view_idx} ({view_tag}): "
                f"answer={cand['answer']!r}, info_visible={cand['info_visible']}, "
                f"reason={cand['reason']!r}"
            )

            # If this is the last view, no further crops
            if view_idx == num_views - 1:
                break

            # ---------- 2) CROP policy for NEXT view ----------
            policy_instr = self._build_policy_instruction(
                question=question,
                view_tag=view_tag,
                view_index=view_idx,
                total_views=num_views,
            )
            raw_policy = self._deepseek_call(current_img, policy_instr)
            print("\n[agent] ---- FULL ASSISTANT OUTPUT (POLICY) ----")
            print(raw_policy)
            print("[agent] ----------------------------------------")

            policy_obj = self._parse_policy_line(raw_policy)
            print(f"[agent] Parsed policy at view {view_idx}: {policy_obj}")

            if policy_obj["action"] != "CROP":
                history.append({
                    "action": "POLICY_ERROR",
                    "view": view_tag,
                    "reason": policy_obj.get("reason", ""),
                    "raw_line": policy_obj.get("raw_line", ""),
                })
                print("[agent] Policy error; stopping further crops.")
                break

            base_bbox = policy_obj["bbox"]
            # Expand with margin
            bbox = self._expand_bbox_with_margin(base_bbox)

            zone = policy_obj.get("zone", "?")
            reason = policy_obj.get("reason", "")

            # Avoid degenerate loop: if new crop is nearly identical to last, stop
            if last_crop_bbox is not None:
                diff = sum(abs(a - b) for a, b in zip(bbox, last_crop_bbox))
                if diff < crop_dup_epsilon:
                    history.append({
                        "action": "CROP_DUP",
                        "from_view": view_tag,
                        "zone": zone,
                        "bbox": bbox,
                        "reason": reason,
                    })
                    print("[agent] New CROP bbox almost identical to previous; "
                          "avoiding infinite loop; stopping further crops.")
                    break

            history.append({
                "action": "CROP",
                "from_view": view_tag,
                "to_view": f"crop_{view_idx+1}",
                "zone": zone,
                "bbox": bbox,
                "reason": reason,
            })
            print(f"[agent] View {view_idx}: CROP_ZONE {zone} bbox={bbox} reason={reason}")

            # Apply crop → next view
            current_img = crop_normalized(current_img, bbox)
            last_crop_bbox = bbox

            if self.save_dir is not None:
                crop_path = os.path.join(self.save_dir, f"step_{view_idx:02d}_crop.png")
                current_img.save(crop_path)
                print(f"[agent] Saved view-{view_idx} cropped image to {crop_path}")

        # ---------- Aggregation: last-YES-wins with first-candidate fallback ----------
        print("\n[agent] ===== ALL ANSWER CANDIDATES =====")
        for cand in answer_candidates:
            print(
                f"  step={cand['step']}, view={cand['view']}, "
                f"info_visible={cand['info_visible']}, answer={cand['answer']!r}, "
                f"reason={cand['reason']!r}"
            )
        print("[agent] ==================================")

        if not answer_candidates:
            print("[agent] No answer candidates collected; returning None.")
            return {
                "history": history,
                "final_answer": None,
                "answer_candidates": [],
            }

        # Find indices where info_visible is clearly YES-like
        yes_indices = [
            i for i, c in enumerate(answer_candidates)
            if c.get("info_visible", "UNKNOWN") in ("YES", "TRUE", "VISIBLE")
        ]

        if yes_indices:
            best_idx = yes_indices[-1]   # last YES wins
        else:
            best_idx = 0  # fallback: first candidate when everything is NO/UNKNOWN

        best_cand = answer_candidates[best_idx]
        final_answer = best_cand["answer"]

        print(
            f"[agent] Selected final candidate: step={best_cand['step']}, "
            f"view={best_cand['view']}, info_visible={best_cand['info_visible']}, "
            f"answer={final_answer!r}"
        )

        return {
            "history": history,
            "final_answer": final_answer,
            "answer_candidates": answer_candidates,
        }