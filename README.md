# ASSN
Semi-Supervised Portrait Matting via the Collaboration of
Teacher–Student Network and Adaptive Strategies

## News

December 7, 2022: codes are released 🔥  

December 6, 2022: Paper accepted  🎉  

## Abstract
In the portrait matting domain, existing methods rely entirely on annotated images for learning. However, delicate manual annotations are time-consuming and there are few detailed datasets available. To reduce complete dependency on labeled datasets, we design a semi-supervised network (ASSN) with two kinds of innovative adaptive strategies for portrait matting. Three pivotal sub-modules are embedded in our architecture, including a static teacher network (S-TN), a static student network (S-SN), and an adaptive student network (A-SN). 
S-TN and S-SN are modules that need to be trained with a small number of high-quality labeled datasets. Moreover, A-SN and S-SN share the same module parameters.  
When processing unlabeled datasets, A-SN adopts the adaptive strategies designed by us to discard the dependence on labeled datasets. The adaptive strategies include: (i) An auxiliary adaption: The teacher network with complicated design not only provides alpha mattes for the adaptive student network but also transmits rough segmentation results and edge graphs as optimization reference standards. (ii) A self-adjusting adaption: The adaptive network can make self-supervised to the characteristics of different layers. In addition, we have produced a finely annotated dataset for scholars in the field.
Compared with existing datasets, our dataset complements the following two types of data neglected in previous datasets: (i) Images taken by multiple people. (ii) Images under low light conditions.}
