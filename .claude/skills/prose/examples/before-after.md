# What prose mode looks like in practice

These examples show the same information written the default AI way vs the prose way. Use these to calibrate when generating responses.

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

TypeScript's main value is not what it catches — it's what it makes possible. Once you have types, your IDE can tell you what a function expects, what it returns, and what you'll break if you rename a field. In a large codebase that means refactoring goes from a terrifying grep session to something you can actually trust. That's the real payoff, not the compile-time errors.

The friction is real though, and it concentrates at the beginning. Getting `tsconfig.json` right, fighting the type system when you're doing something it doesn't naturally express, explaining to a new contributor why their `any` is a problem — all of that costs time. For a quick script or a prototype you'll throw away in a week, it's straightforwardly not worth it.

The threshold question is roughly this: will multiple people touch this code over multiple months? If yes, TypeScript. If you're the only one, and it's small, JavaScript is fine and there's no reason to pretend otherwise.
