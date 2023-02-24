# Meta-hAOX
# Code for "Meta-hAOX: in silico prediction of metabolic reaction catalyzed by human aldehyde oxidase" this paper  

## Requipments  
* python 3.7
* pyTorch 1.5.0+
* DGL 0.5.2+
* dgllife 
* scikit learn
* rdkit
* numpy  
* 
## fingerprint-based methods 
We applied RDKit to generate atom environment fingerprint to construct SOM prediction models for hAOX .  




## graph-based methods   
The GNN model automatically extracts atom environment features through convolutional layers.  


## sequence-based methods 
We also tried to construct a sequnce to sequence model to predict the SOMs by hAOX


