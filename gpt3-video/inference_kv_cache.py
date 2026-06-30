"""Manim animation explaining transformer inference and KV caching.

Story it tells, in five sections:
  1. Title.
  2. A forward pass produces one logit row per token position, but generation
     only ever uses the LAST row. Every other row is computed and thrown away.
  3. Naively regenerating token-by-token recomputes the Keys/Values of every
     past token at every step -- a triangular pile of redundant work.
  4. The Keys/Values of past tokens never change, so we cache them. Each new
     step computes K/V for exactly ONE new token and reuses the rest.
  5. Summary.

Render (low quality, fast):
    manim -pql inference_kv_cache.py InferenceKVCache
Render (high quality):
    manim -qh inference_kv_cache.py InferenceKVCache

Uses only Text (Pango) mobjects, so no LaTeX installation is required.
"""

from manim import (
    Scene,
    VGroup,
    RoundedRectangle,
    Square,
    Rectangle,
    Text,
    Arrow,
    Brace,
    SurroundingRectangle,
    FadeIn,
    FadeOut,
    Write,
    Create,
    GrowArrow,
    Indicate,
    DOWN,
    UP,
    LEFT,
    RIGHT,
    ORIGIN,
    BLUE,
    BLUE_E,
    PURPLE,
    TEAL,
    GREEN,
    ORANGE,
    RED,
    GOLD,
    GREY,
    WHITE,
    YELLOW,
)

# ---- shared palette ---------------------------------------------------------
TOKEN_COLOR = BLUE
MODEL_COLOR = PURPLE
LOGIT_COLOR = TEAL
USED_COLOR = GOLD
DISCARD_COLOR = GREY
KEY_COLOR = ORANGE
VALUE_COLOR = GREEN
QUERY_COLOR = RED


class InferenceKVCache(Scene):
    def construct(self):
        self.section_title()
        self.section_redundant_logits()
        self.section_generation_waste()
        self.section_kv_cache()
        self.section_summary()

    # -- helpers --------------------------------------------------------------
    def token_box(self, label, color=TOKEN_COLOR, width=1.25, height=0.7):
        """A rounded rectangle with a centered text label."""
        box = RoundedRectangle(
            corner_radius=0.12, width=width, height=height,
            color=color, fill_color=color, fill_opacity=0.18,
        )
        txt = Text(label, font_size=24).move_to(box)
        return VGroup(box, txt)

    def cell(self, color, side=0.42, opacity=0.85):
        return Square(side_length=side, color=color,
                      fill_color=color, fill_opacity=opacity, stroke_width=1.5)

    def logit_row(self, n_cols=8, color=LOGIT_COLOR):
        """A horizontal strip of cells representing a distribution over vocab."""
        cells = VGroup(*[self.cell(color) for _ in range(n_cols)])
        cells.arrange(RIGHT, buff=0.05)
        return cells

    def section_header(self, text):
        header = Text(text, font_size=30, color=WHITE).to_edge(UP, buff=0.4)
        self.play(FadeIn(header, shift=DOWN * 0.2))
        return header

    def clear_all(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects])

    # -- 1. title -------------------------------------------------------------
    def section_title(self):
        title = Text("Transformer Inference", font_size=52, color=WHITE)
        subtitle = Text("KV Caching & the Redundant Logits",
                        font_size=30, color=TEAL)
        subtitle.next_to(title, DOWN, buff=0.35)
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(subtitle))

    # -- 2. only the last logit row is used -----------------------------------
    def section_redundant_logits(self):
        self.section_header("A forward pass scores EVERY position")

        # input tokens
        labels = ["The", "cat", "sat", "on"]
        tokens = VGroup(*[self.token_box(l) for l in labels])
        tokens.arrange(RIGHT, buff=0.3).shift(UP * 1.9)
        self.play(FadeIn(tokens, shift=DOWN * 0.2, lag_ratio=0.15))

        # the model block
        model = Rectangle(width=6.2, height=0.9, color=MODEL_COLOR,
                          fill_color=MODEL_COLOR, fill_opacity=0.2)
        model_label = Text("Transformer (12 blocks)", font_size=24).move_to(model)
        model_grp = VGroup(model, model_label).next_to(tokens, DOWN, buff=0.5)

        in_arrows = VGroup(*[
            Arrow(t.get_bottom(), model.get_top(), buff=0.1,
                  stroke_width=3, max_tip_length_to_length_ratio=0.15)
            for t in tokens
        ])
        self.play(GrowArrow(in_arrows[0]), *[GrowArrow(a) for a in in_arrows[1:]])
        self.play(FadeIn(model_grp))

        # logits: one row per position
        rows = VGroup(*[self.logit_row() for _ in labels])
        rows.arrange(DOWN, buff=0.16).next_to(model_grp, DOWN, buff=0.6)
        row_tags = VGroup(*[
            Text(f"pos {i}  ({labels[i]})", font_size=18, color=GREY)
            for i in range(len(labels))
        ])
        for tag, row in zip(row_tags, rows):
            tag.next_to(row, LEFT, buff=0.25)

        out_arrow = Arrow(model.get_bottom(), rows.get_top(), buff=0.15,
                          stroke_width=3)
        self.play(GrowArrow(out_arrow))
        self.play(Create(rows, lag_ratio=0.05), FadeIn(row_tags))

        caption = Text("logits:  one probability row per token position",
                       font_size=22, color=LOGIT_COLOR)
        caption.next_to(rows, DOWN, buff=0.4)
        self.play(FadeIn(caption))
        self.wait(0.8)

        # highlight: only the last row matters
        last_row = rows[-1]
        keep_box = SurroundingRectangle(last_row, color=USED_COLOR, buff=0.08)
        self.play(
            last_row.animate.set_color(USED_COLOR),
            Create(keep_box),
        )
        used_label = Text("only this row predicts the next token  ->  \"the\"",
                          font_size=22, color=USED_COLOR)
        used_label.next_to(caption, DOWN, buff=0.25)
        self.play(FadeIn(used_label))
        self.wait(0.6)

        # fade the discarded rows
        discarded = VGroup(*rows[:-1])
        self.play(
            discarded.animate.set_color(DISCARD_COLOR).set_opacity(0.3),
            *[t.animate.set_opacity(0.4) for t in row_tags[:-1]],
        )
        waste_label = Text("the other rows are computed, then thrown away",
                           font_size=20, color=DISCARD_COLOR)
        waste_label.next_to(used_label, DOWN, buff=0.2)
        self.play(FadeIn(waste_label))
        self.wait(1.4)
        self.clear_all()

    # -- 3. the redundant recomputation across steps --------------------------
    def section_generation_waste(self):
        self.section_header("Generating naively recomputes the past every step")

        steps = ["The", "cat", "sat", "on"]
        # Build a triangular layout: step t processes tokens 0..t
        triangle = VGroup()
        rows_by_step = []
        for t in range(len(steps)):
            row = VGroup()
            for j in range(t + 1):
                is_new = (j == t)
                c = self.cell(QUERY_COLOR if is_new else KEY_COLOR,
                              side=0.5, opacity=0.85 if is_new else 0.5)
                lbl = Text(steps[j], font_size=16,
                           color=WHITE).move_to(c)
                row.add(VGroup(c, lbl))
            row.arrange(RIGHT, buff=0.12)
            rows_by_step.append(row)
            triangle.add(row)
        triangle.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        triangle.shift(LEFT * 2.2 + DOWN * 0.3)

        step_tags = VGroup()
        for t, row in enumerate(rows_by_step):
            tag = Text(f"step {t+1}", font_size=20, color=GREY)
            tag.next_to(row, LEFT, buff=0.4)
            step_tags.add(tag)

        legend = VGroup(
            VGroup(self.cell(KEY_COLOR, side=0.32, opacity=0.5),
                   Text("reused past K/V (recomputed!)", font_size=18)).arrange(RIGHT, buff=0.2),
            VGroup(self.cell(QUERY_COLOR, side=0.32),
                   Text("the one new token", font_size=18)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        legend.to_edge(RIGHT, buff=0.6).shift(UP * 0.3)

        self.play(FadeIn(legend))
        for t, (row, tag) in enumerate(zip(rows_by_step, step_tags)):
            self.play(FadeIn(tag, shift=RIGHT * 0.2),
                      Create(row, lag_ratio=0.1), run_time=0.7)
        self.wait(0.5)

        # brace over the redundant region (everything but the diagonal)
        brace = Brace(triangle, DOWN, color=KEY_COLOR)
        brace_text = Text("orange cells repeat identical work each step",
                          font_size=20, color=KEY_COLOR)
        brace_text.next_to(brace, DOWN, buff=0.15)
        self.play(Create(brace), FadeIn(brace_text))
        self.wait(1.4)
        self.clear_all()

    # -- 4. the KV cache ------------------------------------------------------
    def section_kv_cache(self):
        self.section_header("KV cache: past Keys & Values never change -> store them")

        steps = ["The", "cat", "sat", "on"]

        # cache rows for K and V that grow over time
        k_label = Text("K cache", font_size=22, color=KEY_COLOR)
        v_label = Text("V cache", font_size=22, color=VALUE_COLOR)
        k_label.shift(UP * 1.4 + LEFT * 4.5)
        v_label.next_to(k_label, DOWN, buff=0.9).align_to(k_label, LEFT)

        k_cells = VGroup()
        v_cells = VGroup()

        status = Text("", font_size=22, color=WHITE).to_edge(DOWN, buff=0.8)
        self.add(status)
        self.play(FadeIn(k_label), FadeIn(v_label))

        prev_k = None
        prev_v = None
        for t, tok in enumerate(steps):
            # the new query for this step
            q = self.cell(QUERY_COLOR, side=0.6)
            q_lbl = Text(tok, font_size=18, color=WHITE).move_to(q)
            q_grp = VGroup(q, q_lbl).to_edge(RIGHT, buff=1.2).shift(UP * 0.2)
            q_tag = Text("new query", font_size=18, color=QUERY_COLOR)
            q_tag.next_to(q_grp, UP, buff=0.2)

            # newly computed K and V for this token
            new_k = self.cell(KEY_COLOR, side=0.6)
            new_k_lbl = Text(tok, font_size=16, color=WHITE).move_to(new_k)
            new_v = self.cell(VALUE_COLOR, side=0.6)
            new_v_lbl = Text(tok, font_size=16, color=WHITE).move_to(new_v)
            new_k_grp = VGroup(new_k, new_k_lbl)
            new_v_grp = VGroup(new_v, new_v_lbl)

            # position them at the end of the current cache row
            if t == 0:
                new_k_grp.next_to(k_label, RIGHT, buff=0.4)
                new_v_grp.next_to(v_label, RIGHT, buff=0.4)
            else:
                new_k_grp.next_to(prev_k, RIGHT, buff=0.12)
                new_v_grp.next_to(prev_v, RIGHT, buff=0.12)
            prev_k, prev_v = new_k_grp, new_v_grp

            new_status = Text(
                f"step {t+1}: compute K/V for \"{tok}\" only, append to cache",
                font_size=22, color=WHITE,
            ).to_edge(DOWN, buff=0.8)

            self.play(
                FadeIn(q_grp), FadeIn(q_tag),
                status.animate.become(new_status),
                run_time=0.6,
            )
            # the single new K/V flows into the cache
            self.play(FadeIn(new_k_grp, shift=LEFT * 0.3),
                      FadeIn(new_v_grp, shift=LEFT * 0.3), run_time=0.5)
            k_cells.add(new_k_grp)
            v_cells.add(new_v_grp)

            # query attends to ALL cached keys (highlight)
            self.play(Indicate(k_cells, color=USED_COLOR, scale_factor=1.08),
                      run_time=0.6)
            self.play(FadeOut(q_grp), FadeOut(q_tag), run_time=0.3)

        punch = Text("one new column per step  ->  O(N) instead of O(N²)",
                     font_size=26, color=USED_COLOR)
        punch.to_edge(DOWN, buff=0.8)
        self.play(status.animate.become(punch))
        self.wait(1.6)
        self.clear_all()

    # -- 5. summary -----------------------------------------------------------
    def section_summary(self):
        title = Text("Why inference caches K and V", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.8)
        self.play(Write(title))

        points = [
            ("A forward pass scores every position,", LOGIT_COLOR),
            ("but generation only needs the LAST logit row.", USED_COLOR),
            ("Past tokens' Keys & Values never change,", KEY_COLOR),
            ("so cache them and compute K/V for one new token per step.", VALUE_COLOR),
            ("Result: each new token costs O(N), not O(N²).", USED_COLOR),
        ]
        lines = VGroup(*[
            Text(f"•  {txt}", font_size=26, color=color)
            for txt, color in points
        ])
        lines.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        lines.next_to(title, DOWN, buff=0.7)

        for line in lines:
            self.play(FadeIn(line, shift=RIGHT * 0.3), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(title), FadeOut(lines))
