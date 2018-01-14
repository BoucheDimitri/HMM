# Projet de Monte-Carlo et méthodes séquentielles #

## Mode d'emploi ##

Pour lancer les algorithmes, il faut se rendre dans les deux principaux scripts python:

  - "main" pour lancer chaque filtre en exécutant les différentes parties de code suivantes:\
    0/ PACKAGES & PARAMETERS INITIALIZATION & DATA GENERATION\
    1/ BOOTSTRAP FILTER qui donne les positions estimées par le filtre vs. les vraies positions
    2/ RESAMPLE MOVE WITH FIXED TAU idem
    3/ RESAMPLE MOVE WITH ESTIMATION OF TAU qui donne les positions estimées par le filtre vs. les vraies positions, 
    les estimations de tau et la variance de la loi a posteriori sachant les observations
    
  - "boucle_main" pour lancer chaque filtre plusieurs fois.
  En lançant boucle_main, vous aurez Q = 100 itérations de l'algorithme qui vont s'effectuer.
  Puis un graphique des distances comme dans le graphe des résultats va s'afficher.
  Vont également s'afficher le graphe des estimations de la variance de l'estimateur particulaire à chaque étape ainsi que la variance de la loi à posteriori de tau.
