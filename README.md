# ICCEP 2019: Calliope_iterative smart district model
Repository for the model and data employed for the ICCEP 2019 paper: C. Del Pero, F. Leonforte, F. Lombardi, N. Stevanato, J. Barbieri, N. Aste, E. Colombo, Modelling of an integrated multi-energy system for a nearly Zero Energy Smart District, ICCEP 2019

## Overview
The model hosted in this repository consists in an integrated coupling between the energy modelling framework Calliope (For further details about Calliope, see: https://github.com/calliope-project), including the distric loads and both power and heat generation technologies, and a thermodynamic simulation model of the closed-loop water heat network that is used as a heat source for water-to-water heat pumps. 

<img src="https://github.com/SESAM-Polimi/ICCEP-Calliope_iterative/blob/master/System%20configuration%20scheme.jpg" width="600">

The two models are combined by an iterative process:
- the dispatch strategy of the energy system is optimised based on a LP formulation with HPs operating with fixed timeseries of COPs; 
- the timeseries of COPs are, in turns, updated as a result of the optimal dispatch and of the interactions between HPs operation and the heat network, whose temperature over the year changes in response to such interaction;
- the iteration continues until reaching a heat network temperature trend that is not significantly different from that calculated on the previous iteration.

Further details about the methodology are reported in the related publication.

## Calliope version
To run the Calliope model in this repository without conflicts, please use the Calliope "developer" 0.6.3-dev0 version available here: https://github.com/FLomb/calliope

## License
[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-sa/4.0/)

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
