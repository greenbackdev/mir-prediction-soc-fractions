# Predicting the proportion of centennially stable soil organic carbon using mid-infrared spectroscopy

This code accompanies the paper "Predicting the proportion of centennially stable soil organic carbon using mid-infrared spectroscopy".

**Authors:**

Lorenza Pacini<sup>1,2</sup>, Marcus Schiedung<sup>3</sup>, Marija Stojanova<sup>1,4</sup>, Pierre Roudier<sup>5</sup>, Pierre Arbelet<sup>2</sup>, Pierre Barré<sup>1,*</sup>, François Baudin<sup>6</sup>, Aurélie Cambou<sup>7</sup>, Lauric Cécillon<sup>8</sup>, Jussi Heinonsalo<sup>9</sup>, Kristiina Karhu<sup>9</sup>, Sam McNally<sup>5</sup>, Pascal Omondiagbe<sup>5</sup>, Christopher Poeplau<sup>3</sup>, Nicolas P. A. Saby<sup>10</sup>. 

<sup>1</sup>Laboratoire de Géologie, École Normale Supérieure, CNRS, PSL University, IPSL, Paris, France

<sup>2</sup>Greenback (commercial name: Genesis), Paris, France 

<sup>3</sup>Thünen Institute of Climate-Smart Agriculture, Braunschweig, Germany

<sup>4</sup>ENS de Lyon, LGL-TPE, UMR 5276, Universite Claude Bernard Lyon1, UJM Saint-Etienne, CNRS, Lyon, France

<sup>5</sup>Manaaki Whenua – Landcare Research, Te Papaioea / Palmerston North, Aotearoa / New Zealand

<sup>6</sup>ISTeP, Sorbonne Université, CY Univ., CNRS, Paris, France

<sup>7</sup>Eco&Sols, Université de Montpellier, CIRAD, INRAE, IRD, Institut Agro Montpellier, France

<sup>8</sup>Ministère de l’Europe et des affaires étrangères, Ambassade de France au Kénya, Nairobi, Kenya

<sup>9</sup>Department of Forest Sciences, Faculty of Agriculture and Forestry, University of Helsinki, Helsinki, Finland

<sup>10</sup>INRAE, Info&Sols, 45075, Orléans, France

*corresponding author

## Requirements and environment set-up

This code runs with Python 3.9.

For the required Python packages, see `Pipfile`. It is advised to install the packages and their dependencies using `pipenv`.


Install `pipenv`:
```
pip install pipenv
```

Install the required Python packages:

```
pipenv install
```

Activate the virtual environment:

```
pipenv shell
```

## Data

Please contact the authors for access to the raw data. They will be provided as an SQLite database upon reasonable request.

## Note

The code was developed to allow for predicting different targets : the total organic carbon (TOC, in g/kg fines), the quantity of stable carbon (g/kg fines), the quantity of active carbon (g/kg fines), and the proportion of stable carbon (unitless). Only the results for the prediction of the proportion of stable carbon were presented in the papers. 


## Acknowledgments

The FREACS project was funded by the external call of the EJP Soil (ANR-22-SOIL-0001). This work is part of project ALAMOD of the exploratory research program FairCarboN and received government funding managed by the Agence Nationale de la Recherche under the France 2030 program, ANR-22-PEXF-0002. This work was funded by the New Zealand Government to support the objectives of the Global Research Alliance on Agricultural Greenhouse Gases.The RMQS program is funded by the GIS Sol, a scientific interest group involving the French ministries in charge of the environment and of agriculture, INRAE (National Research Institute for Agriculture, Food and the Environment), ADEME (French Agency for Ecological Transition), IRD (French National Institute for Sustainable Development), IGN (National Institute of Geographic and Forest Information), OFB (French Biodiversity Agency) and BRGM (French geological survey). The authors thank the regional partners who collected soil samples and the staff of Info&Sols and of the European Conservatory for Soil Samples (CEES) involved in the RMQS program.