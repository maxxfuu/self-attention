"""In-depth manim animation: building Q/K/V, then PREFILL, then why KV cache.

Phase 1 -- How Q, K, V are computed (the matmul, step by step):
    X (input embeddings, N x d_model)  @  W_Q (d_model x d_k)  =  Q (N x d_k)
    Each input ROW (one token's embedding vector) times the weight matrix
    produces one ROW of Q. We show a single cell as a dot product, then fill
    Q row by row, then note K = X@W_K and V = X@W_V.

Phase 2 -- Prefill (the first inference forward pass over the whole prompt):
    All prompt tokens are pushed through at once, so Q, K, V get every row
    filled in parallel. Attention runs (Q@K^T -> mask -> softmax -> @V), but
    only the LAST row's output predicts the first generated token. The K and V
    matrices are saved into the KV cache.

Phase 3 -- Decode & why the cache matters:
    Generating token-by-token, a cacheless model would recompute K/V for the
    entire prompt every step -- but prefill already computed those. The cache
    keeps them, so each decode step computes K/V for just the one new token.

Render:
    manim -qh prefill.py PrefillKVCache       # high quality
    manim -ql prefill.py PrefillKVCache       # fast draft

Text-only (Pango); no LaTeX required.
"""

from manim import (
    Scene,
    VGroup,
    Square,
    Rectangle,
    RoundedRectangle,
    Text,
    Arrow,
    DashedLine,
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
    UR,
    DL,
    ORIGIN,
    BLUE,
    BLUE_E,
    PURPLE,
    TEAL,
    GREEN,
    ORANGE,
    RED,
    RED_E,
    GOLD,
    GREY,
    WHITE,
    YELLOW,
)

X_COLOR = BLUE
WQ_COLOR = PURPLE
Q_COLOR = RED
K_COLOR = ORANGE
V_COLOR = GREEN
USED_COLOR = GOLD
REDUNDANT_COLOR = RED_E

PROMPT = ["The", "cat", "sat", "on"]


class PrefillKVCache(Scene):
    def construct(self):
        self.section_title()
        self.phase1_matmul()
        self.phase1_kv()
        self.phase2_prefill()
        self.phase3_decode()
        self.section_summary()

    # -- helpers --------------------------------------------------------------
    def grid(self, n_rows, n_cols, color, side=0.4, fill_op=0.16, stroke=1.2):
        """Return (group, cells2d) where cells2d[r][c] is the Square mobject."""
        cells = []
        group = VGroup()
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                sq = Square(side_length=side, color=color, fill_color=color,
                            fill_opacity=fill_op, stroke_width=stroke)
                sq.move_to(RIGHT * c * side + DOWN * r * side)
                row.append(sq)
                group.add(sq)
            cells.append(row)
        group.move_to(ORIGIN)
        return group, cells

    def row_of(self, cells, r):
        return VGroup(*cells[r])

    def col_of(self, cells, c):
        return VGroup(*[cells[r][c] for r in range(len(cells))])

    def strip(self, labels, color, side=0.6, fill_op=0.85, fs=18):
        """A horizontal strip of labelled cells (one per token)."""
        g = VGroup()
        for lab in labels:
            sq = Square(side_length=side, color=color, fill_color=color,
                        fill_opacity=fill_op, stroke_width=1.4)
            t = Text(lab, font_size=fs, color=WHITE).move_to(sq)
            g.add(VGroup(sq, t))
        g.arrange(RIGHT, buff=0.1)
        return g

    def header(self, text, color=WHITE):
        h = Text(text, font_size=30, color=color).to_edge(UP, buff=0.35)
        self.play(FadeIn(h, shift=DOWN * 0.2))
        return h

    def clear_all(self):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects])

    # -- title ----------------------------------------------------------------
    def section_title(self):
        t = Text("Prefill & the KV Cache", font_size=50, color=WHITE)
        s = Text("from the Q/K/V matmul to inference",
                 font_size=26, color=TEAL).next_to(t, DOWN, buff=0.35)
        self.play(Write(t))
        self.play(FadeIn(s, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(t), FadeOut(s))

    # -- phase 1: Q = X @ W_Q step by step ------------------------------------
    def phase1_matmul(self):
        head = self.header("Phase 1 — computing Q = X · W_Q")

        N, d_model, d_k = len(PROMPT), 6, 4
        side = 0.4

        Xg, Xc = self.grid(N, d_model, X_COLOR, side)
        Wg, Wc = self.grid(d_model, d_k, WQ_COLOR, side)
        Qg, Qc = self.grid(N, d_k, Q_COLOR, side, fill_op=0.0)  # start empty

        # classic matmul layout: Q bottom-right, X left of Q, W_Q above Q
        Qg.move_to(RIGHT * 2.3 + DOWN * 1.55)
        Xg.next_to(Qg, LEFT, buff=0.9)
        Wg.next_to(Qg, UP, buff=0.6).align_to(Qg, LEFT)

        x_lbl = Text("X  (token embeddings)", font_size=18, color=X_COLOR).next_to(Xg, DOWN, buff=0.2)
        w_lbl = Text("W_Q", font_size=20, color=WQ_COLOR).next_to(Wg, UP, buff=0.2)
        q_lbl = Text("Q  (queries)", font_size=18, color=Q_COLOR).next_to(Qg, DOWN, buff=0.2)

        # token labels next to each X row
        tok_tags = VGroup(*[
            Text(PROMPT[r], font_size=16, color=GREY).next_to(self.row_of(Xc, r), LEFT, buff=0.25)
            for r in range(N)
        ])

        self.play(FadeIn(Xg), FadeIn(Wg), Create(Qg), FadeIn(x_lbl), FadeIn(w_lbl), FadeIn(q_lbl))
        self.play(FadeIn(tok_tags, lag_ratio=0.1))

        # --- detail: compute one cell Q[0][0] as a dot product ---
        x_row0 = self.row_of(Xc, 0)
        w_col0 = self.col_of(Wc, 0)
        box_x = SurroundingRectangle(x_row0, color=YELLOW, buff=0.04)
        box_w = SurroundingRectangle(w_col0, color=YELLOW, buff=0.04)
        self.play(Create(box_x), Create(box_w))

        eq = Text("Q[0,0]  =  row x₀  ·  column₀(W_Q)  =  Σ over k of  x_k · w_k",
                  font_size=22, color=WHITE).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(eq))
        # fill that single cell
        self.play(Qc[0][0].animate.set_fill(Q_COLOR, opacity=0.85),
                  Flash(Qc[0][0], color=Q_COLOR, flash_radius=0.4))
        self.wait(0.8)

        # fill the rest of row 0
        self.play(box_w.animate.become(SurroundingRectangle(Wg, color=YELLOW, buff=0.05)))
        rest0 = VGroup(*[Qc[0][c] for c in range(1, d_k)])
        self.play(rest0.animate.set_fill(Q_COLOR, opacity=0.85), run_time=0.5)
        note = Text("one input row  ->  one Q row", font_size=22, color=USED_COLOR).to_edge(DOWN, buff=0.5)
        self.play(Transform(eq, note))
        self.wait(0.6)

        # fill remaining rows one at a time
        self.play(FadeOut(box_x), FadeOut(box_w))
        for r in range(1, N):
            bx = SurroundingRectangle(self.row_of(Xc, r), color=YELLOW, buff=0.04)
            self.play(Create(bx), run_time=0.3)
            self.play(self.row_of(Qc, r).animate.set_fill(Q_COLOR, opacity=0.85),
                      Flash(self.row_of(Qc, r), color=Q_COLOR, flash_radius=0.5),
                      run_time=0.45)
            self.play(FadeOut(bx), run_time=0.2)

        done = Text("Q is now full: every token has a query vector",
                    font_size=22, color=Q_COLOR).to_edge(DOWN, buff=0.5)
        self.play(Transform(eq, done))
        self.wait(1.2)

        # stash references for the next sub-section
        self._p1 = dict(Xg=Xg, Xc=Xc, x_lbl=x_lbl, tok_tags=tok_tags,
                        Qg=Qg, eq=eq, Wg=Wg, w_lbl=w_lbl, q_lbl=q_lbl)

    def phase1_kv(self):
        # reuse X; show that K and V come from the SAME X, different weights
        p = self._p1
        self.play(FadeOut(p["eq"]), FadeOut(p["Wg"]), FadeOut(p["w_lbl"]),
                  FadeOut(p["Qg"]), FadeOut(p["q_lbl"]))

        N = len(PROMPT)
        # three output matrices Q, K, V as filled strips of rows
        def filled(color):
            g, c = self.grid(N, 4, color, 0.4, fill_op=0.85)
            return g, c

        Qg, _ = filled(Q_COLOR)
        Kg, _ = filled(K_COLOR)
        Vg, _ = filled(V_COLOR)
        outs = VGroup(Qg, Kg, Vg).arrange(RIGHT, buff=1.1)
        outs.next_to(p["Xg"], RIGHT, buff=1.4)
        VGroup(outs, p["Xg"]).move_to(ORIGIN).shift(DOWN * 0.3)

        ql = Text("Q = X·W_Q", font_size=18, color=Q_COLOR).next_to(Qg, DOWN, buff=0.2)
        kl = Text("K = X·W_K", font_size=18, color=K_COLOR).next_to(Kg, DOWN, buff=0.2)
        vl = Text("V = X·W_V", font_size=18, color=V_COLOR).next_to(Vg, DOWN, buff=0.2)

        arrows = VGroup(
            Arrow(p["Xg"].get_right(), Qg.get_left(), buff=0.15, stroke_width=3),
            Arrow(p["Xg"].get_right(), Kg.get_left(), buff=0.15, stroke_width=3),
            Arrow(p["Xg"].get_right(), Vg.get_left(), buff=0.15, stroke_width=3),
        )
        self.play(*[GrowArrow(a) for a in arrows])
        self.play(FadeIn(Qg), FadeIn(Kg), FadeIn(Vg), FadeIn(ql), FadeIn(kl), FadeIn(vl))
        note = Text("same input X, three weight matrices  ->  Q, K, V",
                    font_size=22, color=USED_COLOR).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(1.6)
        self.clear_all()

    # -- phase 2: prefill -----------------------------------------------------
    def phase2_prefill(self):
        head = self.header("Phase 2 — PREFILL: the whole prompt in one pass", color=TEAL)

        toks = self.strip(PROMPT, X_COLOR).to_edge(UP, buff=1.1)
        plabel = Text("prompt (all tokens at once)", font_size=18, color=GREY).next_to(toks, RIGHT, buff=0.3)
        self.play(FadeIn(toks, lag_ratio=0.1), FadeIn(plabel))

        # Q, K, V all rows filled in parallel
        Q = self.strip(PROMPT, Q_COLOR, side=0.5).shift(UP * 0.2 + LEFT * 3.5)
        K = self.strip(PROMPT, K_COLOR, side=0.5).move_to(UP * 0.2)
        V = self.strip(PROMPT, V_COLOR, side=0.5).shift(UP * 0.2 + RIGHT * 3.5)
        for grp, lab, col in [(Q, "Q", Q_COLOR), (K, "K", K_COLOR), (V, "V", V_COLOR)]:
            t = Text(lab, font_size=20, color=col).next_to(grp, UP, buff=0.15)
            grp.add(t)
        self.play(FadeIn(Q), FadeIn(K), FadeIn(V))
        par = Text("every row computed simultaneously — fully parallel",
                   font_size=20, color=TEAL).next_to(VGroup(Q, K, V), DOWN, buff=0.5)
        self.play(FadeIn(par))
        self.wait(1.0)
        self.play(FadeOut(par))

        # only the last token predicts the first new token
        last_q = Q[-2]  # last cell before the label was added... use explicit:
        last_cell = Q[len(PROMPT) - 1]
        box = SurroundingRectangle(last_cell, color=USED_COLOR, buff=0.06)
        only = Text("only the LAST position's output predicts the next token",
                    font_size=20, color=USED_COLOR).next_to(VGroup(Q, K, V), DOWN, buff=0.5)
        self.play(Create(box), FadeIn(only))
        self.wait(1.2)
        self.play(FadeOut(box), FadeOut(only))

        # K and V are saved to the cache
        cache_box = SurroundingRectangle(VGroup(K, V), color=GOLD, buff=0.25)
        cache_lbl = Text("K and V saved -> KV cache", font_size=24, color=GOLD)
        cache_lbl.next_to(cache_box, DOWN, buff=0.3)
        self.play(Create(cache_box), FadeIn(cache_lbl))
        self.wait(1.6)
        self.clear_all()

    # -- phase 3: decode & redundancy ----------------------------------------
    def phase3_decode(self):
        head = self.header("Phase 3 — DECODE: why the cache is needed")

        gen = ["the", "mat"]            # tokens generated after the prompt
        full = PROMPT + gen

        # cache row, pre-populated by prefill (PROMPT), grows during decode
        cache_lbl = Text("KV cache", font_size=22, color=GOLD).to_edge(LEFT, buff=0.5).shift(UP * 1.6)
        prefill_cells = self.strip(PROMPT, K_COLOR, side=0.55).next_to(cache_lbl, RIGHT, buff=0.4)
        tag = Text("filled by prefill", font_size=16, color=GOLD).next_to(prefill_cells, UP, buff=0.15)
        self.play(FadeIn(cache_lbl), FadeIn(prefill_cells), FadeIn(tag))
        self.wait(0.6)

        # ---- the cacheless redundancy, shown as a growing triangle ----
        sub = Text("without a cache, each step recomputes K/V for the whole sequence",
                   font_size=20, color=REDUNDANT_COLOR).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(sub))

        rows = VGroup()
        top_y = 0.4
        left_x = LEFT * 4.6
        counter_val = 0
        counter = Text("redundant recomputations: 0", font_size=20, color=REDUNDANT_COLOR)
        counter.to_corner(UP + RIGHT, buff=0.5).shift(DOWN * 0.9)
        self.play(FadeIn(counter))

        for step in range(len(gen)):
            seq_len = len(PROMPT) + step + 1     # prompt + already-generated + new
            row = VGroup()
            for pos in range(seq_len):
                is_new = (pos == seq_len - 1)
                color = V_COLOR if is_new else REDUNDANT_COLOR
                op = 0.9 if is_new else 0.5
                sq = Square(side_length=0.5, color=color, fill_color=color,
                            fill_opacity=op, stroke_width=1.2)
                lab = Text(full[pos], font_size=13, color=WHITE).move_to(sq)
                row.add(VGroup(sq, lab))
            row.arrange(RIGHT, buff=0.1)
            row.move_to(left_x + RIGHT * row.width / 2).set_y(top_y - step * 0.7)
            step_tag = Text(f"decode {step+1}", font_size=18, color=GREY).next_to(row, LEFT, buff=0.35)

            redundant = VGroup(*row[: seq_len - 1])
            self.play(FadeIn(step_tag), FadeIn(row, lag_ratio=0.06), run_time=0.6)
            self.play(Indicate(redundant, color=YELLOW, scale_factor=1.1), run_time=0.6)
            counter_val += (seq_len - 1)
            nc = Text(f"redundant recomputations: {counter_val}",
                      font_size=20, color=REDUNDANT_COLOR).to_corner(UP + RIGHT, buff=0.5).shift(DOWN * 0.9)
            self.play(Transform(counter, nc), run_time=0.35)
            rows.add(VGroup(step_tag, row))

        self.wait(0.8)
        # the fix: cache means only the green diagonal is computed
        fix = Text("the cache already holds those — so compute only the new (green) cell",
                   font_size=20, color=USED_COLOR).to_edge(DOWN, buff=0.5)
        self.play(Transform(sub, fix))
        # append the two generated tokens to the cache to show the cached path
        prev = prefill_cells[-1]
        for g in gen:
            newc = VGroup(
                Square(side_length=0.55, color=V_COLOR, fill_color=V_COLOR,
                       fill_opacity=0.9, stroke_width=1.4),
            )
            t = Text(g, font_size=15, color=WHITE).move_to(newc)
            cellg = VGroup(newc, t).next_to(prev, RIGHT, buff=0.1)
            prev = cellg
            self.play(GrowFromEdge(cellg, LEFT), run_time=0.4)
        self.wait(1.6)
        self.clear_all()

    # -- summary --------------------------------------------------------------
    def section_summary(self):
        title = Text("Prefill, decode, and the cache", font_size=36, color=WHITE).to_edge(UP, buff=0.8)
        self.play(Write(title))
        points = [
            ("Q, K, V = X · W:  each token's row times the weights.", Q_COLOR),
            ("Prefill runs every prompt row at once, in parallel.", TEAL),
            ("Prefill stores the prompt's K and V in the cache.", GOLD),
            ("Decode adds one token at a time.", V_COLOR),
            ("Without the cache it re-derives the prompt's K/V every step.", REDUNDANT_COLOR),
            ("With it: one new K/V per step  ->  O(N) not O(N²).", USED_COLOR),
        ]
        lines = VGroup(*[Text(f"•  {t}", font_size=24, color=c) for t, c in points])
        lines.arrange(DOWN, buff=0.32, aligned_edge=LEFT).next_to(title, DOWN, buff=0.6)
        for ln in lines:
            self.play(FadeIn(ln, shift=RIGHT * 0.3), run_time=0.5)
        self.wait(2.4)
        self.play(FadeOut(title), FadeOut(lines))
