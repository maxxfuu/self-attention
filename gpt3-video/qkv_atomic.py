"""Phase 1, ATOMIC: how one row of Q is built, one scalar x weight-row at a time.

This is the *row-space* view of matrix multiplication (the intuitive one):

    q_0 = x_0 @ W_Q
        = x_0[0]*row_0(W_Q) + x_0[1]*row_1(W_Q) + x_0[2]*row_2(W_Q)

i.e. take ONE dimension of the input row (a single scalar), multiply it by the
WHOLE corresponding row of the weight matrix (a vector) -> that gives one
partial vector. Each input dimension produces one partial vector; ADD them and
you get the output row q_0. Do that for every token row and STACK the results
-> the full Q matrix for the whole sequence.

Concrete worked example (verified):
    x_0      = [2, 1, 3]
    W_Q rows = [1,0,2,1] / [0,3,1,2] / [2,1,0,1]
    2*[1,0,2,1] = [2,0,4,2]
    1*[0,3,1,2] = [0,3,1,2]
    3*[2,1,0,1] = [6,3,0,3]
    sum         = [8,6,5,7] = q_0

Render:
    manim -qh qkv_atomic.py QKVAtomic
    manim -ql qkv_atomic.py QKVAtomic

Text-only (Pango); no LaTeX required.
"""

from manim import (
    Scene,
    VGroup,
    Square,
    Text,
    Arrow,
    DashedLine,
    Line,
    Brace,
    SurroundingRectangle,
    FadeIn,
    FadeOut,
    Write,
    Create,
    GrowArrow,
    Indicate,
    Flash,
    Transform,
    TransformFromCopy,
    DOWN,
    UP,
    LEFT,
    RIGHT,
    ORIGIN,
    BLUE,
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

X_COLOR = BLUE
WQ_COLOR = PURPLE
PART_COLOR = TEAL
Q_COLOR = RED
USED_COLOR = GOLD

X0 = [2, 1, 3]
WQ = [[1, 0, 2, 1],
      [0, 3, 1, 2],
      [2, 1, 0, 1]]
TOKENS = ["The", "cat", "sat", "on"]
SIDE = 0.62


class QKVAtomic(Scene):
    def construct(self):
        self.intro()
        self.atomic_partials()
        self.sum_columns()
        self.assemble_Q()
        self.outro()

    # -- helpers --------------------------------------------------------------
    def vcell(self, val, color, side=SIDE, fs=24, fill_op=0.18):
        sq = Square(side_length=side, color=color, fill_color=color,
                    fill_opacity=fill_op, stroke_width=1.6)
        if val is None or val == "":
            return VGroup(sq)
        t = Text(str(val), font_size=fs, color=WHITE).move_to(sq)
        return VGroup(sq, t)

    def vrow(self, vals, color, side=SIDE, fill_op=0.18):
        g = VGroup(*[self.vcell(v, color, side, fill_op=fill_op) for v in vals])
        g.arrange(RIGHT, buff=0.0)
        return g

    def vcol(self, vals, color, side=SIDE, fill_op=0.18):
        g = VGroup(*[self.vcell(v, color, side, fill_op=fill_op) for v in vals])
        g.arrange(DOWN, buff=0.0)
        return g

    def matrix(self, rows_vals, color, side=SIDE, fill_op=0.18):
        rows = VGroup(*[self.vrow(r, color, side, fill_op) for r in rows_vals])
        rows.arrange(DOWN, buff=0.0)
        return rows

    def header(self, text, color=WHITE):
        h = Text(text, font_size=28, color=color).to_edge(UP, buff=0.3)
        self.play(FadeIn(h, shift=DOWN * 0.2))
        return h

    def caption(self, text, color=WHITE, fs=24):
        return Text(text, font_size=fs, color=color).to_edge(DOWN, buff=0.5)

    # -- intro: lay out x0 (column) and W_Q (matrix) --------------------------
    def intro(self):
        self.head = self.header(
            "Phase 1 (atomic): q₀ = a weighted sum of W_Q's rows")

        # x0 as a vertical column on the left; W_Q as a 3x4 matrix beside it,
        # rows aligned so x0[k] sits on the same line as row k of W_Q.
        self.x_col = self.vcol(X0, X_COLOR).shift(LEFT * 5.0 + UP * 0.6)
        self.wq = self.matrix(WQ, WQ_COLOR).next_to(self.x_col, RIGHT, buff=1.4)
        # keep rows vertically aligned with x_col cells
        self.wq.align_to(self.x_col, UP)

        x_lbl = Text("x₀  (token 'The')", font_size=20, color=X_COLOR).next_to(self.x_col, UP, buff=0.25)
        wq_lbl = Text("W_Q  (3 × 4)", font_size=20, color=WQ_COLOR).next_to(self.wq, UP, buff=0.25)
        dim_tags = VGroup(
            Text("dim 0", font_size=15, color=GREY).next_to(self.x_col[0], LEFT, buff=0.2),
            Text("dim 1", font_size=15, color=GREY).next_to(self.x_col[1], LEFT, buff=0.2),
            Text("dim 2", font_size=15, color=GREY).next_to(self.x_col[2], LEFT, buff=0.2),
        )
        self.play(FadeIn(self.x_col), FadeIn(x_lbl), FadeIn(dim_tags))
        self.play(FadeIn(self.wq), FadeIn(wq_lbl))

        idea = self.caption(
            "one input dimension (a scalar)  ×  one whole weight row (a vector)  =  a partial vector",
            color=USED_COLOR, fs=22)
        self.play(FadeIn(idea))
        self.wait(1.4)
        self.play(FadeOut(idea))
        self.static = VGroup(self.x_col, self.wq, x_lbl, wq_lbl, dim_tags)

    # -- the three atomic scalar*row products ---------------------------------
    def atomic_partials(self):
        self.partials = []
        px = 2.6
        for k in range(3):
            scalar = X0[k]
            wrow_vals = WQ[k]
            result = [scalar * w for w in wrow_vals]

            # highlight the scalar (dim k) and the weight row k
            box_x = SurroundingRectangle(self.x_col[k], color=YELLOW, buff=0.05)
            box_w = SurroundingRectangle(self.wq[k], color=YELLOW, buff=0.05)
            self.play(Create(box_x), Create(box_w), run_time=0.5)

            # arithmetic caption
            cap = self.caption(
                f"{scalar} · {wrow_vals}  =  {result}", color=PART_COLOR, fs=26)
            self.play(FadeIn(cap))

            # the partial vector appears to the right, on the SAME line as row k
            p = self.vrow(result, PART_COLOR, fill_op=0.85)
            p.move_to([px, self.wq[k].get_y(), 0])
            ptag = Text(f"x₀[{k}] · row {k}", font_size=15, color=PART_COLOR).next_to(p, RIGHT, buff=0.3)
            self.play(TransformFromCopy(self.wq[k], p), FadeIn(ptag), run_time=0.8)
            self.play(Flash(p, color=PART_COLOR, flash_radius=0.6), run_time=0.4)
            self.partials.append(VGroup(p, ptag))

            self.play(FadeOut(box_x), FadeOut(box_w), FadeOut(cap), run_time=0.4)

        note = self.caption(
            "three input dimensions  ->  three partial vectors", color=PART_COLOR, fs=24)
        self.play(FadeIn(note))
        self.wait(1.2)
        self.play(FadeOut(note))

    # -- sum the partial vectors column-by-column into q0 ---------------------
    def sum_columns(self):
        # drop the per-partial tags, keep just the three partial rows
        prows = [grp[0] for grp in self.partials]
        self.play(*[FadeOut(grp[1]) for grp in self.partials])

        # plus signs between the stacked partials
        plus1 = Text("+", font_size=30, color=WHITE).next_to(prows[1], LEFT, buff=0.2)
        plus2 = Text("+", font_size=30, color=WHITE).next_to(prows[2], LEFT, buff=0.2)
        line = Line(prows[2].get_left() + DOWN * 0.35 + LEFT * 0.1,
                    prows[2].get_right() + DOWN * 0.35, color=WHITE)
        self.play(FadeIn(plus1), FadeIn(plus2), Create(line))

        # empty q0 row beneath the line, columns aligned with the partials
        q_cells = VGroup(*[self.vcell("", Q_COLOR, fill_op=0.0) for _ in range(4)])
        q_cells.arrange(RIGHT, buff=0.0)
        q_cells.next_to(line, DOWN, buff=0.3).align_to(prows[2], LEFT)
        self.play(Create(q_cells))
        q_lbl = Text("q₀", font_size=22, color=Q_COLOR).next_to(q_cells, RIGHT, buff=0.3)
        self.play(FadeIn(q_lbl))

        result = [8, 6, 5, 7]
        for j in range(4):
            col = VGroup(prows[0][j], prows[1][j], prows[2][j])
            terms = f"{WQ[0][j]*X0[0]}+{WQ[1][j]*X0[1]}+{WQ[2][j]*X0[2]}"
            cap = self.caption(f"column {j}:   {terms}  =  {result[j]}",
                               color=USED_COLOR, fs=26)
            self.play(Indicate(col, color=YELLOW, scale_factor=1.15),
                      FadeIn(cap), run_time=0.6)
            num = Text(str(result[j]), font_size=24, color=WHITE).move_to(q_cells[j])
            self.play(q_cells[j].animate.set_fill(Q_COLOR, opacity=0.85),
                      FadeIn(num), run_time=0.4)
            self.play(FadeOut(cap), run_time=0.2)
            q_cells[j].add(num)

        done = self.caption(
            "q₀ = [8, 6, 5, 7]   —  add the partial vectors (it's a SUM, not a concat)",
            color=Q_COLOR, fs=24)
        self.play(FadeIn(done))
        self.wait(1.6)
        # keep q0 around, clear the rest of the working
        self.q_final = VGroup(q_cells, q_lbl)
        fade = [plus1, plus2, line, done] + prows
        self.play(*[FadeOut(m) for m in fade], FadeOut(self.static))
        self.wait(0.2)

    # -- repeat per token, distilled, stack rows -> full Q --------------------
    def assemble_Q(self):
        new_head = Text("Repeat for every token row, then stack → Q",
                        font_size=28, color=WHITE).to_edge(UP, buff=0.3)
        self.play(Transform(self.head, new_head), FadeOut(self.q_final))

        Xvals = [[2, 1, 3], [1, 2, 0], [0, 1, 2], [3, 0, 1]]
        Qvals = [[8, 6, 5, 7], [1, 6, 4, 5], [4, 5, 1, 4], [5, 1, 6, 4]]
        ms = 0.45

        # X (4x3) on the left
        X = self.matrix(Xvals, X_COLOR, side=ms).to_edge(LEFT, buff=0.7).shift(UP * 1.4)
        x_lbl = Text("X  (4×3)", font_size=16, color=X_COLOR).next_to(X, DOWN, buff=0.2)
        tok_tags = VGroup(*[
            Text(TOKENS[r], font_size=14, color=GREY).next_to(X[r], LEFT, buff=0.18)
            for r in range(4)
        ])

        # W_Q in the middle
        wq = self.matrix(WQ, WQ_COLOR, side=ms).next_to(X, RIGHT, buff=0.8)
        wq_lbl = Text("W_Q  (3×4)", font_size=16, color=WQ_COLOR).next_to(wq, DOWN, buff=0.2)

        # Q outline (empty) on the right, rows aligned with X rows
        Qcells = [[self.vcell("", Q_COLOR, side=ms, fill_op=0.0) for _ in range(4)]
                  for _ in range(4)]
        Qrowg = [VGroup(*Qcells[r]).arrange(RIGHT, buff=0.0) for r in range(4)]
        Q = VGroup(*Qrowg).arrange(DOWN, buff=0.0).to_edge(RIGHT, buff=0.9).align_to(X, UP)
        q_lbl = Text("Q  (4×4)", font_size=16, color=Q_COLOR).next_to(Q, DOWN, buff=0.2)

        self.play(FadeIn(X), FadeIn(x_lbl), FadeIn(tok_tags),
                  FadeIn(wq), FadeIn(wq_lbl), FadeIn(Q), FadeIn(q_lbl))

        recipe = self.caption(
            "each token: scale every W_Q row by an input dim, then add → its Q row",
            color=USED_COLOR, fs=20)
        self.play(FadeIn(recipe))

        work_y = -1.45
        for r in range(4):
            # spotlight this token's input row
            bx = SurroundingRectangle(X[r], color=YELLOW, buff=0.04)
            self.play(Create(bx), tok_tags[r].animate.set_color(YELLOW), run_time=0.4)

            # build the three partial vectors (scalar x weight-row)
            prows = [self.vrow([Xvals[r][k] * w for w in WQ[k]], PART_COLOR,
                               side=0.42, fill_op=0.8) for k in range(3)]
            stack = VGroup(*prows).arrange(DOWN, buff=0.0).move_to([0.4, work_y, 0])
            tags = VGroup(*[
                Text(f"×{Xvals[r][k]}", font_size=14, color=GREY).next_to(prows[k], LEFT, buff=0.18)
                for k in range(3)
            ])
            self.play(*[TransformFromCopy(wq[k], prows[k]) for k in range(3)],
                      FadeIn(tags), run_time=0.9)
            self.wait(0.3)

            # ease the multiplier tags FULLY out before the addition appears,
            # so the "x" and "+" never occupy the same spot
            self.play(FadeOut(tags), run_time=0.4)

            # now reveal the addition symbols and the sum line
            plus = VGroup(Text("+", font_size=22).next_to(prows[1], LEFT, buff=0.18),
                          Text("+", font_size=22).next_to(prows[2], LEFT, buff=0.18))
            line = Line(ORIGIN, RIGHT * 1.68, color=WHITE).next_to(stack, DOWN, buff=0.06)
            self.play(FadeIn(plus), Create(line), run_time=0.4)

            result = self.vrow(Qvals[r], PART_COLOR, side=0.42, fill_op=0.85)
            result.next_to(line, DOWN, buff=0.16).align_to(prows[0], LEFT)
            self.play(FadeIn(result), Flash(result, color=PART_COLOR, flash_radius=0.45),
                      run_time=0.5)

            # the result row flies up into Q row r (and recolors to Q's red)
            target = self.vrow(Qvals[r], Q_COLOR, side=ms, fill_op=0.85).move_to(Qrowg[r])
            self.play(Transform(result, target), run_time=0.7)

            # clear the working area for the next token (tags already eased out)
            self.play(FadeOut(bx), FadeOut(stack),
                      FadeOut(plus), FadeOut(line), run_time=0.35)

        self.play(FadeOut(recipe))
        note = self.caption(
            "every token row → one Q row;  stack them → Q for the whole sequence",
            color=USED_COLOR, fs=22)
        self.play(FadeIn(note))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self.head])

    # -- outro: K and V are the same procedure --------------------------------
    def outro(self):
        new_head = Text("K and V: same X, same procedure, different weights",
                        font_size=28, color=WHITE).to_edge(UP, buff=0.3)
        self.play(Transform(self.head, new_head))

        items = VGroup(
            Text("Q = X · W_Q", font_size=30, color=Q_COLOR),
            Text("K = X · W_K", font_size=30, color=ORANGE),
            Text("V = X · W_V", font_size=30, color=GREEN),
        ).arrange(DOWN, buff=0.5).move_to(ORIGIN)
        for it in items:
            self.play(FadeIn(it, shift=RIGHT * 0.3), run_time=0.5)

        kicker = self.caption(
            "each is just: one input dimension × one weight row, summed across dimensions",
            color=USED_COLOR, fs=22)
        self.play(FadeIn(kicker))
        self.wait(2.2)
        self.play(FadeOut(self.head), FadeOut(items), FadeOut(kicker))
