"""Comprehensive manim animation: KV redundancy in autoregressive inference.

The goal is to *literally see* the wasted work. Without a KV cache, every
generation step re-runs the whole sequence, recomputing the Keys and Values of
all previous tokens -- even though those vectors are bit-for-bit identical to
what was computed on earlier steps (a decoder-only token's K/V depend only on
positions <= itself, which never change once produced).

Sections:
  1. Title.
  2. What one token produces: a Key, a Value, a Query.
  3. THE CENTREPIECE -- step-by-step generation with NO cache. Each step the
     past K/V cells flash red ("recomputed AGAIN"); only the diagonal cell is
     new. A counter tallies redundant recomputations as the triangle grows.
  4. The redundancy table: N new vs N(N-1)/2 redundant.
  5. With a KV cache: one new K/V per step, past ones reused -> linear.
  6. Cost comparison: O(N^2) vs O(N).
  7. Summary.

Render:
    manim -qh kv_redundancy.py KVRedundancy      # high quality
    manim -ql kv_redundancy.py KVRedundancy      # fast draft

Text-only (Pango); no LaTeX required.
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
    GrowFromEdge,
    Indicate,
    Flash,
    Transform,
    DOWN,
    UP,
    LEFT,
    RIGHT,
    ORIGIN,
    BLUE,
    PURPLE,
    TEAL,
    GREEN,
    GREEN_E,
    ORANGE,
    RED,
    RED_E,
    GOLD,
    GREY,
    WHITE,
    YELLOW,
)

TOKEN_COLOR = BLUE
KEY_COLOR = ORANGE
VALUE_COLOR = GREEN
QUERY_COLOR = RED
NEW_COLOR = GREEN
REDUNDANT_COLOR = RED_E
USED_COLOR = GOLD

TOKENS = ["The", "cat", "sat", "on", "mat"]


class KVRedundancy(Scene):
    def construct(self):
        self.section_title()
        self.section_one_token()
        self.section_stepwise_redundancy()
        self.section_redundancy_table()
        self.section_with_cache()
        self.section_cost()
        self.section_summary()

    # -- helpers --------------------------------------------------------------
    def kv_cell(self, label, color, side=0.62, opacity=0.85, font_size=18):
        sq = Square(side_length=side, color=color,
                    fill_color=color, fill_opacity=opacity, stroke_width=1.5)
        txt = Text(label, font_size=font_size, color=WHITE).move_to(sq)
        return VGroup(sq, txt)

    def token_box(self, label, color=TOKEN_COLOR, width=1.1, height=0.62):
        box = RoundedRectangle(corner_radius=0.1, width=width, height=height,
                               color=color, fill_color=color, fill_opacity=0.18)
        txt = Text(label, font_size=22).move_to(box)
        return VGroup(box, txt)

    def header(self, text, color=WHITE):
        h = Text(text, font_size=30, color=color).to_edge(UP, buff=0.35)
        self.play(FadeIn(h, shift=DOWN * 0.2))
        return h

    def clear_all(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects])

    # -- 1. title -------------------------------------------------------------
    def section_title(self):
        t = Text("KV Redundancy in Inference", font_size=50, color=WHITE)
        s = Text("watching every generation step recompute the past",
                 font_size=26, color=TEAL).next_to(t, DOWN, buff=0.35)
        self.play(Write(t))
        self.play(FadeIn(s, shift=UP * 0.2))
        self.wait(1.2)
        self.play(FadeOut(t), FadeOut(s))

    # -- 2. what one token produces -------------------------------------------
    def section_one_token(self):
        head = self.header("Each token produces a Key, a Value, and a Query")

        tok = self.token_box("cat").shift(UP * 1.4)
        self.play(FadeIn(tok, shift=DOWN * 0.2))

        k = self.kv_cell("K", KEY_COLOR, side=0.8, font_size=24)
        v = self.kv_cell("V", VALUE_COLOR, side=0.8, font_size=24)
        q = self.kv_cell("Q", QUERY_COLOR, side=0.8, font_size=24)
        kvq = VGroup(k, v, q).arrange(RIGHT, buff=1.1).next_to(tok, DOWN, buff=1.2)

        arrows = VGroup(*[
            Arrow(tok.get_bottom(), m.get_top(), buff=0.12, stroke_width=3)
            for m in kvq
        ])
        self.play(*[GrowArrow(a) for a in arrows])
        self.play(FadeIn(kvq, shift=DOWN * 0.2, lag_ratio=0.2))

        labels = VGroup(
            Text("Key: what I offer", font_size=18, color=KEY_COLOR).next_to(k, DOWN, buff=0.3),
            Text("Value: what I carry", font_size=18, color=VALUE_COLOR).next_to(v, DOWN, buff=0.3),
            Text("Query: what I seek", font_size=18, color=QUERY_COLOR).next_to(q, DOWN, buff=0.3),
        )
        self.play(FadeIn(labels, lag_ratio=0.2))

        note = Text("K and V depend only on this token + its position -> they never change",
                    font_size=22, color=USED_COLOR).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(note))
        self.wait(1.6)
        self.clear_all()

    # -- 3. CENTREPIECE: step-by-step redundancy ------------------------------
    def section_stepwise_redundancy(self):
        head = self.header("No cache: every step recomputes ALL past K/V",
                           color=REDUNDANT_COLOR)

        # legend
        legend = VGroup(
            VGroup(self.kv_cell("", NEW_COLOR, side=0.3),
                   Text("new this step", font_size=18)).arrange(RIGHT, buff=0.18),
            VGroup(self.kv_cell("", REDUNDANT_COLOR, side=0.3),
                   Text("recomputed (redundant)", font_size=18)).arrange(RIGHT, buff=0.18),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        legend.to_corner(UP + RIGHT, buff=0.5).shift(DOWN * 0.5)
        self.play(FadeIn(legend))

        # running counter of redundant recomputations
        counter_val = 0
        counter = Text("redundant K/V recomputed: 0",
                       font_size=24, color=REDUNDANT_COLOR)
        counter.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(counter))

        rows = VGroup()
        prev_row = None
        first_cell_left = LEFT * 4.6  # left anchor for column 0
        top_y = 1.9

        for step in range(len(TOKENS)):       # step index 0..4 -> length step+1
            step_lbl = Text(f"step {step + 1}", font_size=20, color=GREY)

            # build this row's K/V cells (one per position 0..step)
            row = VGroup()
            for pos in range(step + 1):
                is_new = (pos == step)
                color = NEW_COLOR if is_new else REDUNDANT_COLOR
                cell = self.kv_cell(TOKENS[pos], color, side=0.6,
                                    opacity=0.9 if is_new else 0.55)
                row.add(cell)
            row.arrange(RIGHT, buff=0.12)
            # position: align left anchor, stack downward
            row.move_to(first_cell_left + RIGHT * (row.width / 2),
                        ).set_y(top_y - step * 0.78)
            step_lbl.next_to(row, LEFT, buff=0.4)

            # animate: redundant cells appear then FLASH red, new cell pops green
            redundant_cells = VGroup(*row[:step]) if step > 0 else VGroup()
            new_cell = row[step]

            self.play(FadeIn(step_lbl, shift=RIGHT * 0.2),
                      FadeIn(row, lag_ratio=0.08), run_time=0.6)
            if step > 0:
                self.play(Indicate(redundant_cells, color=YELLOW,
                                   scale_factor=1.12), run_time=0.6)
                counter_val += step      # `step` new redundant cells this round
                new_counter = Text(
                    f"redundant K/V recomputed: {counter_val}",
                    font_size=24, color=REDUNDANT_COLOR).to_edge(DOWN, buff=0.5)
                self.play(Transform(counter, new_counter), run_time=0.4)
            # emphasize the single genuinely-new cell
            self.play(Flash(new_cell, color=NEW_COLOR, flash_radius=0.5),
                      run_time=0.4)

            rows.add(VGroup(step_lbl, row))
            prev_row = row

        self.wait(0.4)
        # brace the redundant triangle (all but the diagonal)
        brace = Brace(rows, DOWN, color=REDUNDANT_COLOR)
        brace_txt = Text("10 of these 15 computations are pure repetition",
                         font_size=22, color=REDUNDANT_COLOR)
        brace_txt.next_to(brace, DOWN, buff=0.15)
        self.play(Create(brace), FadeIn(brace_txt))
        self.wait(1.8)
        self.clear_all()

    # -- 4. the count ---------------------------------------------------------
    def section_redundancy_table(self):
        head = self.header("Count it: only the diagonal is real work")

        n = len(TOKENS)
        grid = VGroup()
        cells_by_rc = {}
        for r in range(n):
            for c in range(n):
                if c < r:
                    color, op = REDUNDANT_COLOR, 0.55     # below diagonal: redundant
                elif c == r:
                    color, op = NEW_COLOR, 0.9            # diagonal: new
                else:
                    color, op = GREY, 0.12               # above: doesn't exist (future)
                cell = Square(side_length=0.7, color=color,
                              fill_color=color, fill_opacity=op, stroke_width=1.2)
                cell.move_to(RIGHT * c * 0.8 + DOWN * r * 0.8)
                cells_by_rc[(r, c)] = cell
                grid.add(cell)
        grid.move_to(ORIGIN).shift(DOWN * 0.2 + LEFT * 1.2)

        col_tags = VGroup(*[
            Text(TOKENS[c], font_size=16, color=GREY).move_to(
                cells_by_rc[(0, c)]).next_to(cells_by_rc[(0, c)], UP, buff=0.15)
            for c in range(n)
        ])
        row_tags = VGroup(*[
            Text(f"step {r+1}", font_size=16, color=GREY).next_to(
                cells_by_rc[(r, 0)], LEFT, buff=0.3)
            for r in range(n)
        ])
        self.play(Create(grid, lag_ratio=0.02), FadeIn(col_tags), FadeIn(row_tags))

        tally = VGroup(
            Text(f"new (diagonal):  {n}", font_size=24, color=NEW_COLOR),
            Text(f"redundant (below):  {n*(n-1)//2}", font_size=24, color=REDUNDANT_COLOR),
            Text(f"total computed:  {n*(n+1)//2}", font_size=24, color=WHITE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        tally.to_edge(RIGHT, buff=0.7)
        self.play(FadeIn(tally, lag_ratio=0.25))
        self.wait(1.8)
        self.clear_all()

    # -- 5. with cache --------------------------------------------------------
    def section_with_cache(self):
        head = self.header("With a KV cache: compute each K/V exactly once",
                           color=NEW_COLOR)

        k_label = Text("K cache", font_size=22, color=KEY_COLOR)
        v_label = Text("V cache", font_size=22, color=VALUE_COLOR)
        k_label.to_edge(LEFT, buff=0.6).shift(UP * 1.0)
        v_label.next_to(k_label, DOWN, buff=1.0).align_to(k_label, LEFT)

        self.play(FadeIn(k_label), FadeIn(v_label))

        counter = Text("total K/V computed: 0", font_size=24, color=NEW_COLOR)
        counter.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(counter))

        k_cells, v_cells = VGroup(), VGroup()
        prev_k = prev_v = None
        for step, tok in enumerate(TOKENS):
            q = self.kv_cell(tok, QUERY_COLOR, side=0.62)
            q.to_edge(RIGHT, buff=1.4).shift(UP * 0.1)
            q_tag = Text("new query", font_size=16, color=QUERY_COLOR).next_to(q, UP, buff=0.15)

            new_k = self.kv_cell(tok, KEY_COLOR, side=0.62)
            new_v = self.kv_cell(tok, VALUE_COLOR, side=0.62)
            if step == 0:
                new_k.next_to(k_label, RIGHT, buff=0.4)
                new_v.next_to(v_label, RIGHT, buff=0.4)
            else:
                new_k.next_to(prev_k, RIGHT, buff=0.12)
                new_v.next_to(prev_v, RIGHT, buff=0.12)
            prev_k, prev_v = new_k, new_v

            self.play(FadeIn(q, shift=LEFT * 0.2), FadeIn(q_tag), run_time=0.4)
            self.play(GrowFromEdge(new_k, LEFT), GrowFromEdge(new_v, LEFT),
                      run_time=0.45)
            k_cells.add(new_k)
            v_cells.add(new_v)
            # query reads the WHOLE cache (reused, not recomputed)
            self.play(Indicate(k_cells, color=USED_COLOR, scale_factor=1.06),
                      run_time=0.5)
            new_counter = Text(f"total K/V computed: {step + 1}",
                               font_size=24, color=NEW_COLOR).to_edge(DOWN, buff=0.6)
            self.play(Transform(counter, new_counter),
                      FadeOut(q), FadeOut(q_tag), run_time=0.35)

        punch = Text("one new K/V per step  ->  N total, zero repetition",
                     font_size=26, color=USED_COLOR).to_edge(DOWN, buff=0.6)
        self.play(Transform(counter, punch))
        self.wait(1.6)
        self.clear_all()

    # -- 6. cost comparison ---------------------------------------------------
    def section_cost(self):
        head = self.header("The cost: quadratic vs linear")

        n = len(TOKENS)
        no_cache_total = n * (n + 1) // 2   # 15
        cache_total = n                     # 5
        unit = 0.32

        # baseline
        base_y = -2.2
        # without-cache bar
        nc_bar = Rectangle(width=1.6, height=no_cache_total * unit,
                           color=REDUNDANT_COLOR, fill_color=REDUNDANT_COLOR,
                           fill_opacity=0.6)
        nc_bar.move_to(LEFT * 2.5).align_to([0, base_y, 0], DOWN)
        nc_lbl = Text("no cache", font_size=22, color=REDUNDANT_COLOR).next_to(nc_bar, DOWN, buff=0.2)
        nc_val = Text(f"{no_cache_total}  ~  O(N²)", font_size=22, color=REDUNDANT_COLOR).next_to(nc_bar, UP, buff=0.2)

        # with-cache bar
        c_bar = Rectangle(width=1.6, height=cache_total * unit,
                          color=NEW_COLOR, fill_color=NEW_COLOR, fill_opacity=0.6)
        c_bar.move_to(RIGHT * 2.5).align_to([0, base_y, 0], DOWN)
        c_lbl = Text("KV cache", font_size=22, color=NEW_COLOR).next_to(c_bar, DOWN, buff=0.2)
        c_val = Text(f"{cache_total}  ~  O(N)", font_size=22, color=NEW_COLOR).next_to(c_bar, UP, buff=0.2)

        self.play(GrowFromEdge(nc_bar, DOWN), GrowFromEdge(c_bar, DOWN), run_time=1.2)
        self.play(FadeIn(nc_lbl), FadeIn(nc_val), FadeIn(c_lbl), FadeIn(c_val))

        note = Text("K/V vectors computed to generate N tokens",
                    font_size=22, color=WHITE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(1.8)
        self.clear_all()

    # -- 7. summary -----------------------------------------------------------
    def section_summary(self):
        title = Text("KV caching, in one breath", font_size=36, color=WHITE).to_edge(UP, buff=0.8)
        self.play(Write(title))
        points = [
            ("A past token's Key and Value never change.", KEY_COLOR),
            ("Without a cache, every step recomputes them all.", REDUNDANT_COLOR),
            ("That is N(N-1)/2 wasted recomputations.", REDUNDANT_COLOR),
            ("Cache them, compute one new K/V per step.", VALUE_COLOR),
            ("Generation drops from O(N²) to O(N).", USED_COLOR),
        ]
        lines = VGroup(*[Text(f"•  {t}", font_size=26, color=c) for t, c in points])
        lines.arrange(DOWN, buff=0.38, aligned_edge=LEFT).next_to(title, DOWN, buff=0.7)
        for ln in lines:
            self.play(FadeIn(ln, shift=RIGHT * 0.3), run_time=0.55)
        self.wait(2.2)
        self.play(FadeOut(title), FadeOut(lines))
