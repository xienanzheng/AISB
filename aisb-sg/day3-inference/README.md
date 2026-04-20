# [Pranav] Day 3: Model deployment / inference (model day)

- [ ] *Training pipeline*
      * *Good to think about all the stages*
- [ ] Jailbreaks
- [ ] Tokenization vulnerabilities
- [ ] Guardrails: Constitutional classifiers and linear probes for input and output monitoring
      * (input + output) Another llm that you ask yes/no
      * Break this
      * Another llm reasons, then says yes/no
      * Break this
      * Obviously, this does not stream. How to stream
      * So, output classifier with linear probe?
      * Reading: constitutional classifier
- [ ] Stretch
      * Adversarial examples and attacks on image models
      * Watermarking techniques and detection
- [ ] Model weight extraction attacks (distillation, SVD)
      * (pranav) Teach them the training loop with distillation
        - [ ] Also watermarking and detecting distillation from model
      * SVD
      * Stretch: read more papers on distillation - subliminal learning,
