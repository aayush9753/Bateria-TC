# What prose mode looks like in practice

These examples show the same information written the default AI way vs the prose way. Use these to calibrate when generating responses.

**Critical principle: prose mode means the same information in better form — not less information.** A prose version that drops technical details to sound conversational has failed. Every specific point from the "default" version should survive into the prose version; the prose just connects them into argument rather than fragmenting them into a list.

---

## Summarizing a research paper

### DEFAULT (do not write like this)

**Summary of "Attention Is All You Need" (Vaswani et al., 2017)**

**Key Contributions:**
- Introduced the Transformer architecture
- Eliminated recurrence and convolutions entirely
- Used self-attention mechanisms for sequence modeling

**Main Ideas:**
- **Multi-head attention:** Allows the model to attend to different positions simultaneously
- **Positional encoding:** Injects sequence order information since there's no recurrence
- **Encoder-decoder structure:** Standard for translation tasks

**Results:**
- Achieved state-of-the-art on WMT 2014 English-to-German translation
- 28.4 BLEU score, outperforming previous models
- Significantly faster training than RNN-based models

**Why it matters:** Became the foundation for BERT, GPT, and essentially all modern large language models.

---

### PROSE (write like this)

The paper's central bet is that attention alone — no convolutions, no recurrence — is sufficient for sequence modeling, and the bet paid off badly enough to embarrass everyone who thought otherwise. The Transformer processes the entire input at once rather than step by step, which lets it learn long-range dependencies that RNNs kept losing across many tokens. To replace the sequence information that recurrence would have provided naturally, they inject sinusoidal positional encodings into the embeddings — a slightly awkward patch that works well enough that no one has found a compelling reason to replace it.

The multi-head attention mechanism is where most of the action is. Rather than a single attention operation, they run several in parallel, each learning to attend to a different kind of relationship. One head might track syntactic agreement, another might handle coreference — they don't specify it, but that's roughly what probing experiments later found. The encoder-decoder structure is fairly standard for translation; the interesting architecture is the full attention stack underneath it.

The results were decisive. 28.4 BLEU on WMT 2014 English-to-German, faster to train than previous state-of-the-art, and much more parallelizable. In hindsight the more important result was that it became the substrate for everything that followed — BERT took the encoder, GPT took the decoder, and the field more or less stopped building anything else.

---

## Explaining a technical concept

### DEFAULT (do not write like this)

**What is gradient descent?**

Gradient descent is an optimization algorithm used to minimize a loss function. Here's how it works:

1. **Initialize weights** randomly
2. **Compute the loss** using the current weights
3. **Calculate gradients** of the loss with respect to each weight
4. **Update weights** in the direction that reduces loss:
   - `weight = weight - learning_rate × gradient`
5. **Repeat** until convergence

**Key variants:**
- **Batch gradient descent:** Uses the full dataset each step
- **Stochastic gradient descent (SGD):** Uses one example at a time
- **Mini-batch:** Uses a small batch (most common in practice)

**Why it works:** Moving opposite to the gradient always moves you toward lower loss values.

---

### PROSE (write like this)

Gradient descent is a way of asking: given where I am in the loss landscape, which direction is downhill? You compute the gradient — the slope of the loss function with respect to each weight — and then nudge every weight slightly in the direction that makes the slope negative. Do that enough times and you find a minimum, or at least something low enough to be useful.

The "learning rate" is how big each nudge is, and it matters more than people expect. Too large and you overshoot and oscillate; too small and you converge so slowly that training takes weeks. There's no principled way to pick it in advance, which is why a significant fraction of practical deep learning is fiddling with learning rate schedules.

In practice almost nobody uses true gradient descent, where you'd compute the gradient over the entire dataset before taking a step. You use mini-batches — small random subsets, typically 32 or 256 examples — and treat the gradient over that batch as an estimate of the true gradient. It's noisier, but the noise turns out to help by preventing the optimizer from getting stuck in shallow local minima. The stochasticity is a feature disguised as a compromise.

---

## Giving an opinion or analysis

### DEFAULT (do not write like this)

**Thoughts on using TypeScript vs JavaScript:**

**Pros of TypeScript:**
- Static typing catches errors at compile time
- Better IDE support and autocomplete
- Easier refactoring for large codebases
- Enforces code contracts through interfaces

**Cons of TypeScript:**
- Additional compilation step required
- Learning curve for type system
- Can add boilerplate with complex generics
- Configuration overhead (tsconfig.json)

**Recommendation:** Use TypeScript for large projects or teams. JavaScript is fine for small scripts or prototypes.

---

### PROSE (write like this)

Note: this version covers every technical point from the DEFAULT — static typing, autocomplete, refactoring, interfaces, compilation step, learning curve, generics, tsconfig — but in connected argument rather than a disconnected list.

TypeScript's real value isn't the compile-time error catches, though those are real. It's that the type system makes information available throughout the development workflow rather than only at runtime. Your IDE can autocomplete a method call because it knows the return type of the previous call. It can warn you that you're passing a string where a number is expected before you even run anything. And the interfaces don't just help the tooling — they're documentation that the compiler enforces, which means a function's contract is stated once and can't silently drift from its implementation.

The cost side is more specific than "learning curve." There's a compilation step that adds friction to local development, and it's heavier at the start: configuring `tsconfig.json` to match what your project actually needs takes time, and if you inherit a messy one, untangling it is its own project. The type system itself stays manageable for most everyday code, but it has corners — conditional types, `infer`, higher-kinded patterns — where you end up writing type-level code nearly as complex as the runtime code it's describing. And when someone hits that wall and reaches for `any` to escape, you've lost the guarantee without the compiler noticing, which is the most insidious failure mode.

The threshold that makes this worth it: is this code that multiple people will read and modify over months? That's the scenario TypeScript is built for. Most of its value is communication — to future you, to colleagues, to the IDE acting as a translator between what you wrote and what everything else expects. A one-off script you write alone and run twice gets almost none of that benefit while paying the full configuration cost. For that, JavaScript is fine and there's no honest reason to pretend otherwise.
