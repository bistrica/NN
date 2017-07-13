# NN

<b>Uwaga: należy do klasy LexicalUnit w PLWNGraphBuilder/vertices/lexical_unit.py dodać pole self.polarity.</b>

Należy rozpocząć od pliku parser i na samym dole pliku wprowadzić ścieżkę do pliku config.txt zgodnym z przykładowym configiem.

Instrukcja do configa:


Opis postępowania

Korzystać z programu można na następujące sposoby:
1. Połączyć graf będący wyjściem z mergera PLWNBuilder (wersja 2.) z grafem jednostek leksykalnych i zapisać go na dysku.
2. Całkowicie połączony, zapisany na dysku graf graf poddać propagacji.
3. Wykonać punkty 1. i 2. bez zapisywania grafu na dysku (niepolecane).

Uwaga! Dla każdej operacji należy wprowadzić dane bazy danych (zmienne HOST, DB_NAME, USER, PASS).

1. Łączenie grafów

W celu propagacji, należy najpierw połączyć graf będący wyjściem z mergera PLWNBuilder (wersja 2.) z grafem jednostek leksykalnych. Można zapisać go na dysku i użyć potem do propagacji lub bezpośrednio użyć do propagacji bez zapisu na dysku (niezalecane).


By połączyć grafy, należy ponadto:
1. Wprowadzić ścieżkę grafu połączonego przez PLWNBuilder (zmienna MERGED_PATH)
2. Wprowadzić ścieżkę grafu jednostek leksykalnych (zmienna LU_PATH)
3. Wprowadzić id relacji, które chcemy uwzględnić w nowym grafie (lista idków lub wartość 'all' dla zmiennej RELS_TO_APPEND). Ponadto dodane będą relacje synonimii o wartości -8.
4. Można wprowadzić wartość dla zmiennej SAVE_MERGED_GRAPH_PATH, aby zapisać połączony graf wynikowy.
5. Zmienne HOST, DB_NAME, USER, PASS muszą być uzupełnione.
====


2. Propagacja 

Propagacji trzeba dokonać z użyciem połączonego w pełni grafu.
Sposób 1. Uzupełnij dane z punktu 1. Wówczas graf zostanie stworzony (i ewentualnie zapisany) przed propagacją.
Sposób 2. Uzupełnij jedynie zmienną LU_PATH, która zawierać będzie ścieżkę do w pełni połączonego grafu zgodnie z punktem 1.
Zmienne HOST, DB_NAME, USER, PASS muszą być uzupełnione.

Ponadto należy wybrać sposób i parametry wybranej propagacji.

Metody propagacji to:
1. manualna (TYPE:MANUAL)
2. sieć neuronowa (TYPE:NEURAL)
3. kilka sieci neuronowych, po jednej na każdą część mowy (TYPE:NEURAL_MULTIPLE)
4. klasyfikator k najbliższych sąsiadów (TYPE:KNN)
5. klasyfikator SVM (TYPE:SVM)
6. klasyfikator Bayesa (TYPE:BAYES)
7. zespół klasyfikatorów (TYPE:ENSEMBLE)


2.1. Manualna propagacja

0. Wprowadź wartość zmiennej TYPE:MANUAL.
1. Wprowadź wartość zmiennej MANUAL_RELATION_TYPES. Są to relacje, które będą uzwględniane w propagacji (muszą znajdować się w grafie).
2. Wprowadź wartość zmiennej MANUAL_RELATION_WEIGHTS. Są to wagi odpowiadające wybranym relacjom.
3. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
4. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
5. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
6. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
7. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)

2.2. Sieć neuronowa
0. Wprowadź wartość zmiennej TYPE:NEURAL.
1. Wprowadź wartość zmiennej MANUAL_RELATION_TYPES. Są to relacje, które będą uzwględniane w propagacji (muszą znajdować się w grafie).
3. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
4. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
5. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
6. Wybierz głębokość uczenia sieci. Jeśli sieć uczy się tylko na oznaczonych danych, głębość = 1. Jeśli chcemy uczyć sieć także na danych świeżo rozpropagowanych (np. na oznaczonych przez nią najbliższych sąsiadach), wybierz większą liczbę (zmienna TRAINING_DEPTH).
7. Wybierz liczbę warstw ukrytych i liczbę neuronów w każdej z warstw (zmienna LAYERS_UNITS).
8. Jeśli chcesz zapisać model sieci, uzupełnij zmienną SAVE_NEURAL_NETWORK_MODEL_PATH.
9. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
10. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)

Jeśli zapisałeś model sieci neuronowej, możesz użyć go ponownie. Wówczas:
0. Wprowadź wartość zmiennej TYPE:NEURAL.
1. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
2. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
3. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
4. Wprowadź ścieżkę do pliku, w którym zapisany jest model sieci (zmienna NEURAL_NETWORK_MODEL_PATH).
5. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
6. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)

Przy korzystaniu z zapisanego modelu wyuczonej sieci, zmienna TRAINING_DEPTH nie będzie brana pod uwagę.


2.3. Kilka sieci neuronowych

Sieci te mają taką samą architekturę. Każda z nich odpowiada osobnej części mowy.

0. Wprowadź wartość zmiennej TYPE:NEURAL_MULTIPLE.
1. Wprowadź wybrane id części mowy, dla których chcesz propagować polaryzację (zmienna CHOSEN_POS).
2. Wprowadź wartość zmiennej MANUAL_RELATION_TYPES. Są to relacje, które będą uzwględniane w propagacji (muszą znajdować się w grafie).
3. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
4. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
5. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
6. Wybierz głębokość uczenia sieci. Jeśli sieć uczy się tylko na oznaczonych danych, głębość = 1. Jeśli chcemy uczyć sieć także na danych świeżo rozpropagowanych (np. na oznaczonych przez nią najbliższych sąsiadach), wybierz większą liczbę (zmienna TRAINING_DEPTH).
7. Wybierz liczbę warstw ukrytych i liczbę neuronów w każdej z warstw (zmienna LAYERS_UNITS_NM).
8. Jeśli chcesz zapisać model sieci, uzupełnij zmienną SAVE_NEURAL_NETWORK_MODEL_PATH.
9. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
10. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)

Jeśli zapisałeś modele sieci neuronowej, możesz użyć je ponownie. Wówczas:
0. Wprowadź wartość zmiennej TYPE:NEURAL_MULTIPLE.
1. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
2. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
3. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
4. Wprowadź ścieżkę do pliku, w którym zapisane są modele sieci (zmienna NEURAL_NETWORK_MODEL_PATH).
5. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
6. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)
Przy korzystaniu z zapisanego modelu wyuczonych sieci, zmienna TRAINING_DEPTH nie będzie brana pod uwagę.

2.4. K najbliższych sąsiadów

0. Wprowadź wartość zmiennej TYPE:KNN.
1. Wprowadź wartość zmiennej MANUAL_RELATION_TYPES. Są to relacje, które będą uzwględniane w propagacji (muszą znajdować się w grafie).
3. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
4. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
5. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
6. Wybierz głębokość uczenia modelu. Jeśli model uczy się tylko na oznaczonych danych, głębość = 1. Jeśli chcemy uczyć model także na danych świeżo rozpropagowanych (np. na oznaczonych przez nią najbliższych sąsiadach), wybierz większą liczbę (zmienna TRAINING_DEPTH).
7. Wybierz liczbę sąsiadów (zmienna KNN_NEIGHBOURS_NUMBER).
8. Wybierz algorytm dla klasyfikatora (zmienna KNN_ALGORITHM).
9. Wybierz sposób liczenia wag dla sąsiadów (zmienna KNN_WEIGHTS).
10. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
11. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)

2.5. SVM


0. Wprowadź wartość zmiennej TYPE:SVM.
1. Wprowadź wartość zmiennej MANUAL_RELATION_TYPES. Są to relacje, które będą uzwględniane w propagacji (muszą znajdować się w grafie).
3. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
4. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
5. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
6. Wybierz głębokość uczenia modelu. Jeśli model uczy się tylko na oznaczonych danych, głębość = 1. Jeśli chcemy uczyć model także na danych świeżo rozpropagowanych (np. na oznaczonych przez nią najbliższych sąsiadach), wybierz większą liczbę (zmienna TRAINING_DEPTH).
7. Wybierz rodzaj kernela (zmienna SVM_KERNEL)
8. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
9. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)
10. Jeśli chcesz, możesz zapisać wyuczony model SVM z rozszerzeniem .pkl (zmienna SAVE_SVM_MODEL_PATH).

Jeśli zapisałeś wyuczony model SVM, możesz go użyć ponownie. W tym celu:

0. Wprowadź wartość zmiennej TYPE:SVM.
1. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
2. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
3. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
4. Wprowadź ścieżkę do pliku z rozszerzeniem .pkl, w którym zapisany jest zespół klasyfikatorów (zmienna SVM_MODEL_PATH).
5. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
6. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)


2.6. Naiwny Bayes

0. Wprowadź wartość zmiennej TYPE:BAYES.
1. Wprowadź wartość zmiennej MANUAL_RELATION_TYPES. Są to relacje, które będą uzwględniane w propagacji (muszą znajdować się w grafie).
3. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
4. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
5. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
6. Wybierz głębokość uczenia modelu. Jeśli model uczy się tylko na oznaczonych danych, głębość = 1. Jeśli chcemy uczyć model także na danych świeżo rozpropagowanych (np. na oznaczonych przez nią najbliższych sąsiadach), wybierz większą liczbę (zmienna TRAINING_DEPTH).
7. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
8. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)

2.7 Zespół klasyfikatorów

W skład zespołu wchodzi SVM, zestaw sieci neuronowych (jedna sieć na osobną część mowy) i pojedyncza sieć neuronowa. Zespół zwraca tylko te wyniki, które były klasifkowane przez każdy klasyfikator i dla których jest przynajmniej częściowo zgodny. Znaczy to, że jeśli zestaw sieci neuronowych (tak jak w TYPE:NEURAL_MULTIPLE) klasyfikuje tylko rzeczowniki i przymiotniki, nie zostaną uwzględnione wyniki dla czasowników. Częściowa zgodność z kolei to brak sytuacji, w której jeden z klasyfikatorów nadaje polaryzację pozytywną, a inny - negatywną. Klasyfikatory nie muszą być zgodne w 100%, tzn. różne stopnie dla tej samej polaryzacji lub obecność polaryzacji neutralnej i negatywnej lub pozytywnej nie dyskwalifikują wyniku (wygrywa częstsza wartość).

  
0. Wprowadź wartość zmiennej TYPE:ENSEMBLE.
1. Wprowadź wartość zmiennej MANUAL_RELATION_TYPES. Są to relacje, które będą uzwględniane w propagacji (muszą znajdować się w grafie).
2. Wprowadź wybrane id części mowy, dla których chcesz propagować polaryzację (zmienna CHOSEN_POS).
3. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
4. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
5. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
6. Wybierz rodzaj kernela (zmienna SVM_KERNEL)
7. Wybierz liczbę warstw ukrytych i liczbę neuronów w każdej z warstw dla sieci neuronowej klasyfikującej wszystkie dane (zmienna LAYERS_UNITS) oraz dla sieci neuronowych klasyfikujących osobno każdą część mowy (zmienna LAYERS_UNITS_NM).
8. Wybierz głębokość uczenia modelu i sieci. Jeśli model/sieć uczy się tylko na oznaczonych danych, głębość = 1. Jeśli chcemy uczyć model/sieć także na danych świeżo rozpropagowanych (np. na oznaczonych przez nią najbliższych sąsiadach), wybierz większą liczbę (zmienna TRAINING_DEPTH).
7. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
8. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)
9. Jeśli chcesz, zapisz zespół klasyfikatorów z rozszerzeniem .pkl (zmienna SAVE_ENSEMBLE_PATH)

Jeśli zapisałeś zespół klasyfikatorów, możesz go użyć ponownie. Wówczas:

0. Wprowadź wartość zmiennej TYPE:ENSEMBLE.
1. Wybierz, czy dane mają być normalizowane (zmienna NORMALIZATION).
2. Wybierz głębokość sąsiedztwa, na jaką będą propagowane dane z użyciem uprzednio wyznaczonych wartości polaryzacji sąsiadów (zmienna DEPTH).
3. Wybierz minimalny procent liczby oznaczonych sąsiadów. Jeśli za mało sąsiadów jest oznaczonych, nie propaguje się polaryzacji do węzła (zmienna PERCENT).
4. Wprowadź ścieżkę do pliku z rozszerzeniem .pkl, w którym zapisany jest zespół klasyfikatorów (zmienna ENSEMBLE_PATH).
5. Jeśli chcesz, zapisz graf z nowymi wartościami polaryzacji (zmienna SAVE_MODIFIED_MERGED_GRAPH_PATH)
6. Jeśli chcesz, zapisz do pliku informacje o jednostkach leksykalnych, które uległy propagacji wraz z ich nowymi wartościami polaryzacji (zmienna FILE_LEX_UNITS_WITH_NEW_POLARITY)
Przy korzystaniu z zapisanego zespołu, zmienne TRAINING_DEPTH, SAVE_NEURAL_NETWORK_MODEL_PATH, NEURAL_NETWORK_MODEL_PATH, SVM_MODEL_PATH, SAVE_SVM_MODEL_PATH nie będą brane pod uwagę.


===

3. Klasyfikacja

W przypadku propagacji wyniki nie muszą być oczywiście zapisane w bazie lub grafie, ale otrzymywane są jedynie te, które znajdują się w sąsiedztwie oznaczonych węzłów (również w dalszym, zależnie od wybranej odległości sąsiedztwa). Zdarza się jednak konieczność klasyfikacji dowolnego węzła na podstawie wyuczonego modelu, nawet jeśli nie znajduje się on w sąsiedztwie oznaczonych węzłów i nie spełnia kryterium procentu oznaczonych sąsiadów. (Można propagować dane dla minimalnego procenta oznaczonych sąsiadów równego 0, jednakże wciąż będą to dane blisko danych oznaczonych).

Aby sprawdzić działanie wybranego klasyfikatora, ensemble'a lub ręcznej propagacji na dowolnym węźle/węzłach, należy:
0. Wprowadź dane do klasyfikatora zgodnie z wybranym podpunktem w punkcie 2.
1. Wprowadź wartość zmiennej CLASSIFY_DATA, która stanowi listę wartości id jednostek leksykalnych, które chcesz sklasyfikować.
2. Wprowadź wartość zmiennej FILE_LEX_UNITS_WITH_NEW_POLARITY, która w tym przypadku stanowi ścieżkę do pliku ze sklasifikowanymi jednostkami leksykalnymi.

Zwracana jest wartość None w sytuacji, gdy rekord nie został sklasyfikowany z powodu braku krawędzi (tak, istnieją takie, np. id 12168) lub z powodu trudności metody w podjęciu decyzji (głównie dotyczy ensemble'a i ręcznej propagacji).

Z użyciem tej opcji blokowany jest zapis do bazy (SAVE_TO_DB) i zmodyfikowanego grafu (SAVE_MODIFIED_MERGED_GRAPH_PATH).

 
