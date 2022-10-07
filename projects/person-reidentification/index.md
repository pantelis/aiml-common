# Deep Person Reidentification

## Omniscale Relations (40 points)

Now that you understand the approach that can reveal relationships in synthetic still images lets look at a more complex scenario that involves scenes with people.  

First learn how you can represent humans as a vector. There have been many successful attempts to perform person reidentification at multiple scales such as [this](https://arxiv.org/pdf/1905.00953.pdf] and [this](https://arxiv.org/pdf/1910.06827.pdf). 


The key idea is simple: when we look at other people we use a hierarchical mechanism to determine their id. Over longer time scales people change clothing, hairstyles etc.  Over shorter time scales they move in and out of scenes in many instances with facial features hidden or parts of their body occluded. If we are unable to extract their id we resort to a logical reasoning. 


Your task is to replace the RN baseline with OSNet 


The extra points below can only be used to maximize your project grade and cannot be transferred to other course grades (exams, assignments etc.) 
