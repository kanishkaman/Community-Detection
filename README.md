# Community-Detection
This repository contains my work on community detection using the Girvan-Newman and Louvain algorithms. The project focuses on analyzing two datasets, `lastfm_asia_edges.csv` and `wiki-vote.txt`, to detect communities within networks and visualize the results through dendrograms.
 
## Project Structure
 
- `Assignment1.py`: Contains all the code for running both the Girvan-Newman and Louvain algorithms on the two datasets (`lastfm_asia_edges.csv` and `wiki-vote.txt`). This script covers:
  - Girvan-Newman algorithm on `lastfm_asia_edges.csv`
  - Girvan-Newman algorithm on `wiki-vote.txt`
  - Louvain algorithm on `lastfm_asia_edges.csv`
  - Louvain algorithm on `wiki-vote.txt`
 
- `data/`: This folder contains the datasets used for the analysis.
  - `lastfm_asia_edges.csv`: Dataset from LastFM Asia for community detection.
  - `wiki-vote.txt`: Wiki-Vote dataset used for detecting communities.
 
- `GVmod.txt`: Contains data for removed edges and modularity changes for visualization purposes.
 
- `Plots`: Dendrogram plots generated from the algorithms are also attached.
  - `Plot1.png`: Dendrogram plot for the LastFM Asia dataset.
  - `Plot2.png`: Dendrogram plot for the Wiki-Vote dataset.
 
- `Report1.pdf`: This file is the final report detailing the methodology, results, and analysis of the community detection algorithms applied to the datasets.
