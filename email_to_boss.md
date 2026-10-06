Subject: Paper-inspired coherent meta-optical flow cytometry prototype

Dear [Boss's Name],

I have prepared a short update on our coherent meta-optical flow cytometry prototype, following the design principles described by Behrens, Li, and Monticone in *What should a linear optical frontend compute?*

The current simulation uses three paired phase-only metasurface stations, coherent angular-spectrum propagation, and a 4 × 4 intensity detector at each station. The optical frontend and a lightweight linear classifier are optimized jointly. On the present held-out synthetic phase-cell dataset, the model reaches 100% accuracy after eight training epochs.

The attached slides emphasize the optical geometry and detector assumptions, including which detector parameters are already defined in the model and which still need to be specified for an experimental implementation. I have also separated the current proof-of-concept result from the next validation steps; the reported accuracy is on deliberately separable synthetic cells and should not yet be interpreted as biological classification performance.

The next priorities I propose are:

- replacing the synthetic phase objects with measured cell phase maps;
- adding detector noise, quantization, saturation, and timing constraints;
- comparing fixed optics, M2-only optimization, and joint M1+M2 optimization; and
- evaluating fabrication and alignment tolerances.

I would especially appreciate your feedback on the detector specifications and the most relevant experimental constraints to include in the next iteration.

Best regards,

[Your Name]
