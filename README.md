# Tensor-BC-VARETA:

**Scientific Context**
Magnetoencephalography and Electroencephalography (MEEG) constitute noninvasive techniques that encode information about Neural processes with a good temporal resolution, in the form Scalp electromagnetic Signals. Unraveling Brain Function patterns from these Signal, in the context of a single Subject or Population, involves the methodological stride of reconstructing the Neural Activity and Connectivity. Nevertheless, due to the MEEG Signals low spatial resolution, such reconstruction is distressed by mathematical indetermination, contextually referred as MEEG Inverse Problem (MEEG IP). As consequence, transiting the MEEG IP solution turns to search for Population Brain function patterns is highly sensitive to multiple factors, such as specific Subject Brain Anatomy, Head Conductivity properties, Biological and Instrumentation Noisy signals, and not least unavoidable Idealizations concerning the mathematical modeling. A common agreement has been settled, that informing the MEEG IP Methods with Priors on the spatial structure of Neural Activity and Connectivity, is the only viable resource to bring up the spatial resolution at a Level descriptive of Brain Function. Yet it is argued that those Priors should encode Spatial-Temporal-Populational information of the. The “how” poses controversy still. 

**Methodology**
This software package contains full Bayesian System Identification approach in order to adrees this problem. This is done by the definition of the Frequency Domain MEEG Forward Model in the context of Populations. The Model allows to assimilate data from both modalities EEG or MEG, that may be defined into different Recording Systems. The EEG or MEG Lead Fields correspond to the individual Subjects Head Media conductivity Model and Recording System (Fuchs et al., 2002). The Gray Matter generators, for all Subjects in the Population, are defined into individual surfaces that are Homeomorphic to a common population space (Valdés-Hernández et al. 2009). The Classes within the Population are defined by Covariables (in general Age, Gender, Healthy/Pathological). We represent the Data and Neural Activity of the Class MEEG by a Covariance Components Model (CCM) (Rubin, 1977). The CCM is merged with a Joint Gaussian Graphical Model (JGGM) (Witten et al., 2014) at the Sources Level, where the Subjects’ Connectivity is defined through its individual Precision Matrix. This allows to model the Covariables (Class features) and Spatial information (Short and Long-Range Connections) into the JGGM hyperparameters, specific for the Class or Pairs of Gray Matter Regions. The estimation is tackled into the Expectation Maximization scheme for Maximum Posterior analysis (McLachlan and Krishnan, 2007). This is done by a modification of the original strategy to this Populational case and the Local Quadratic Approximation of the Sources JGGM, as an extension of the formulation for the Maximization step presented in (Paz-Linares and Gonzalez Moreira et al., 2018). 

Tensor-BC-VARETA toolbox is supported by the following publication: 
**Gonzalez-Moreira, E., Paz-Linares, D., Martinez-Montes, E., Valdes-Hernandez, P., Bosch-Bayard, J., Bringas-Vega, ML., Valdes-Sosa, P., (2018), "Populational Super-Resolution Sparse M/EEG Sources and Connectivity Estimation", bioRxiv, 346569.**

Authors:
Pedro A. Valdes-Sosa, 
Deirel Paz-Linares, 
Eduardo Gonzalez-Moreira
