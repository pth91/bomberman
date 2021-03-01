# bomberman_rl

Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Project Participants

* Nils Schacknat [email](mailto:bu225@uni-heidelberg.de)
* Robin Spriegel [email](mailto:dw230@uni-heidelberg.de)
* Philipp Holzträger [email](mailto:holztraeger@stud.uni-heidelberg.de)

## Possible Approaches

* Es sollte so sein, dass wir nur einen diskreten space sowohl was die States als auch die Handlungen angeht.
    * Ich sehe jedenfalls keine weiteren Möglichkeiten, sodass man irgendwas stetiges hätte.

* Hier sind ein paar links zu approaches, die ich zu diskreten räumen auf wikipedia gefunden habe:
    1. [QL/DQN](https://en.wikipedia.org/wiki/Q-learning)
        * Q-Learning und damit Deep Q Network, das scheint die grundlegende theoretische Beschreibung des ganzen als Markovketten zu sein.
    2. [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method)
        * ich denke hier geht es darum die Wkeiten zu finden, mit der bestimmte states eintreten und daraus dann bestimmte aktionen abzuleiten, die moeglichst _gewinnbringend_ sind im sinne vom reinforcement learning.
    3. [SARSA](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action)
        * State-Action-Reward-state-action, ist ein algorithmus zum lernen von markov ketten.
    4. [TD](https://en.wikipedia.org/wiki/Temporal_difference_learning)
        * Temporal difference learning, das scheint das interessanteste zu sein, ohne mit neuronalen netzen anfangen zu muessen... das muesste man denke ich nur mal schauen, ob man das irgendwo so implementiert bekommt mit den vorhandenen bibliotheken
    5. [Github Project pommerman](https://github.com/eugene/pommerman)

## ToDos

1. Wir sollten uns nochmal die Videos der Vorlesung ansehen und schauen, was wir da noch finden könnten.
2. Ich glaube wir kommen nicht drum rum uns zu ueberlegen wie wir die states definieren und wie wir sie bewerten, ich kann mir nicht vorstellen, dass man da irgendwie automatisch sinnvolle regeln finden kann.
    * also zum beispiel, dass wir entweder punkte sammeln wollen oder andere gegenspieler in die luft jagen wollen. und abhaenging von der bewertung dieser ereignisse den agent dann trainieren
