# featuretools

1. featuretools braucht dataframes als grundlage für die sogenannten entities
2. entities sind verschiedene (teil-)aspekte der daten, die man herausstellen will
    * das ist dann sowas wie ein kunde, ein produkt
3. relationships zwischen entities
    * also 1 kund kann n produkte kaufen
4. featuretools erstellt aus entities und deren relationships automatisch neue features
    * ein `EntitySet` ist eine kollektion von entities und deren realationship

## ToDo

1. dataframes mit daten generieren
    * vermutlich ist das dann sowas wie agent positionen, bombe gelegt/nicht gelegt
2. sinnvolle primitves (funktionen in feature tools, die man benutzt um transformationen auf features anzuwenden) ueberlegen und implementieren
    * zum beispiel linearkombinationen von entfernungen der agens zueinander
3. entities und deren relations ueberlegen
    * das koennte schwierig werden, weil man sich dafuer ueberlegen muss wie man die agents (und eventuell informationen ueber das spielfeld) sinnvoll modelliert
    * dafuer gibt es in `featuretools` aber die Area Variable, mit der man mit hilfe von tupeln laengen und breitengrade angeben kann, das heißt man eigentlich schon mal einen anfang um abstände der agents untereinander (oder nur von unserem zu den anderen agents) zu definieren mit hilfe von primitves
    * man muesste aber unabhaengig von `featuretools` das spiel laufen lassen um ueberhaupt erstmal daten zu sammeln, die man sinnvoll dann in den dataframes nutzen kann
4. die generierten features mit hilfe eines ml algos in das qlearning reinpacken
    * das ist vermutlich auch nochmal einiges an aufwand, der gemacht werden muss, damit man am ende etwas bekmomt das lauffähig ist...
