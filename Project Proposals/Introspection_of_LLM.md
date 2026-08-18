# Can an LM distinguish between endogenous and exogenous representations?

A paper entitled [Emergent Introspective Awareness in Large Language Models](https://transformer-circuits.pub/2025/introspection/index.html) found that LLMs can reason about their internal states. They can access and report on these states. The study also found that LLMs can detect whether a concept has been introduced via a steering vector. While these findings were mostly unreliable, the question remains as to the level at which the LM can detect the perturbation via a steering vector. By breaking down the task of detecting a perturbation into knowing about a perturbation and detecting it, we can better understand the model's capabilities and potentially improve them in the future.

In our project, we aim to assess the level at which an LM can recognise and distinguish between endogenous and exogenous representations.
To do this, we will perform two runs:

- a **normal run**, where we prompt the model to think strongly about a concept, e.g. 'France'; and
- an **injected run**, where we give the model an unrelated prompt but inject a vector that produces an activation representation that is extremely similar to the natural concept, e.g. 'France'.

Then we want to assess whether we can find information in the model on how much the model notices the perturbation.
To do this, we will compare the activations of the normal run vs the injected run via multiple methods:
$$h_{natural}≈h_{injected}$$

1. Ask the model if it detected the perturbation.
2. Train a binary probe classifier from the activations.
3. Use the J-Space to evaluate whether a perturbation was detected (list of words).
4. Use NLAs to further examine the activations from the model.