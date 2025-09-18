/*
          CZY POŁOŻENIE GEOGRAFICZNE WPŁYWA NA LICZBĘ WYPADKÓW DROGOWYCH W USA ?
-----------------------------------------------------------------------------------------------

Wstępna analiza danych w środowisku PostgreSQL 

    1. ZBIÓR DANYCH
    
Ogólnokrajowy zbiór danych o wypadkach samochodowych w USA. Dane gromadzone od lutego 2016 r. do grudnia 2021 r. 
przy użyciu interfejsów API, zapewniających strumieniowe przesyłanie danycho zdarzeniach drogowych. 
Interfejsy API transmitują dane o ruchu przechwycone przez różne podmioty tj.: amerykańskie i stanowe 
departamenty transportu, organy ścigania, kamery drogowe i czujniki ruchu w sieciach drogowych.

	2. ZMIENNE:

    • 10 zmiennych dotyczących położenia geograficznego, analizowanych pod kątem wpływ na liczbę wypadków drogowych: 
	id, severity, start_lat, start_lng, end_lat, end_lng, street, city, county, state

    • ilość obserwacji: 2 845 342
    
	Dane jakościowe: id, street, city, county, state

	Dane ilościowe: severity, start_lat, start_lng, end_lat, end_lng


	3. ANALIZA JEDNOCZYNNIKOWA */

--liczba wartości dla zmiennej 'id' (ilosc wypadkow): 2 845 342
select count(*) from us_accidents 

--liczba wartości dla zmiennej 'state": 49
select distinct state from us_accidents ua2 

--liczba wartości dla zmiennej 'city': 11 682
select count(city) from 
	(select distinct city from us_accidents ua2) foo
	
--liczba wartości dla zmiennej 'county': 1 707
select count(county) from 
	(select distinct county from us_accidents ua2) foo

--liczba wartości dla zmiennej 'street': 159 652
select count(street) from 
	(select distinct street from us_accidents ua2) foo
	
--liczba wartości dla zmiennej 'start_lat': 969 629
select count(start_lat) from 
	(select distinct start_lat from us_accidents ua2) foo
	
--liczba wartości dla zmiennej'start_lng': 918407
select count(start_lng) from 
	(select distinct start_lng from us_accidents ua2) foo

--liczba wartości dla zmiennej 'end_lat':954 559
select count(end_lat) from 
	(select distinct end_lat from us_accidents ua2) foo
	
--liczba wartości dla zmiennej 'end_lng':904 741
select count(end_lng) from 
	(select distinct end_lng from us_accidents ua2) foo
	
--liczba wartości dla zmiennej 'serverity':4
select count(severity) from 
	(select distinct severity from us_accidents ua2) foo

/*Liczba wartości przyjmowanych w zbiorze danych dla poszczególnych zmiennych:
    • state: 49
    • city: 11 682
    • county: 1707
    • street: 159 652
    • id: 2 845 342
    • severity: 4
    • start_lat: 969 629
    • start_lng: 918 407
    • end_lat: 954 559
    • end_lng: 904 741


STATYSTYKI PODSUMOWUJĄCE DOTYCZĄCE LICZBY WYPADKÓW DLA ZMIENNEJ 'state'*/

--widok dla wypadkow w poszczegolnych stanach
create view v_wypadki_stany as
select distinct a.state,
count (a.id) ilosc_wypadkow
from us_accidents a 
group by (a.state)
order by ilosc_wypadkow desc;

select ua."Number"::numeric  from us_accidents ua 


select "Number" from us_accidents 

select count(ua."Number") from us_accidents ua 
select count(ua."Number") from us_accidents ua where ua."Number" like ' '

--ilosc wypadków w poszczegolnych stanach z podsumowaniem
select distinct a.state,
count (a.id) ilosc_wypadkow
from us_accidents a 
group by cube (a.state)
order by ilosc_wypadkow desc

--ranking ilosci wypadkow w poszczegolnych stanach
select state, ilosc_wypadkow,
rank () over (order by ilosc_wypadkow desc) ranking
from v_wypadki_stany
order by ilosc_wypadkow desc

--podstawowe funkcje statystyczne dla ilosci wypadkow w poszczegolnych stanach
select 
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow)) wariancja
from v_wypadki_stany

--tablea liczby stanów, w których ilość wypadków przekracza warości średnie
select * from 
(
select '1' as lp, 'obserwacje odstające' as stany, count(*) as ilosc from v_wypadki_stany where ilosc_wypadkow > 122667
union
select '2', 'stany z iloscią wypadków powyzej sredniej', count(*) from v_wypadki_stany where ilosc_wypadkow >58068
union
select '3', 'stany z iloscią wypadków  powyzej mody', count(*) from v_wypadki_stany where ilosc_wypadkow >201
) foo
order by lp

/*
    • rozkład danych (kwantyle, kwartyle, IQR - środkowa połowa zbioru danych):
        ◦ pierwszy kwartyl (Q1) – 25% obserwacji położonych jest poniżej, a 75% powyżej:  6392
        ◦ drugi kwartyl (Q2) (mediana) dzieli zbiór obserwacji na dwie równe części: 20850
        ◦ trzeci kwartyl (Q3) – 75% obserwacji położona jest poniżej, a 25% powyżej: 52 902
        ◦ IQR -środkowa połowa zbioru danych: 46 510
        ◦ granica obserwacji odstających z góry (upper whisker): 122 667
        ◦ kwantyl_10: 2 258
        ◦ kwantyl_90: 113 535
    • tendencja centralna (średnia, mediana, moda), 
        ◦ średnia: 58 068
        ◦ moda (dominująca ilość wypadków): 201
        ◦ mediana: 20850
    • dyspersja
        ◦ min: 201
        ◦ max: 795 868
        ◦ rozstęp danych: 795 667
        ◦ odchylenie standardowe:125 597

Wniosek: Zbiór obserwacji jest bardzo zróżnicowany. Wysokie wartości odchylenia standardowego świadczą o silnym rozproszeniu wartości wokół średniej.
Do wartości odstających z góry można zaliczyć cztery stany: OR,TX, FL,CA.*/


/*STATYSTYKI PODSUMOWUJĄCE DOTYCZĄCE LICZBY WYPADKÓW DLA ZMIENNEJ 'city'*/

--tabela z ilością wypadkow dla poszczególnych miast
select distinct a.state, a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a 
group by a.city, a.state 
order by ilosc_wypadkow desc 


--teren zabudowny, (niepusty numer przy ulicy)
select * from us_accidents ua 

select distinct a.state, a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a 
where a."Number" is not null
group by a.city, a.state 
order by ilosc_wypadkow desc 

--tabela z ilością wypadkow dla poszczególnych miast pogrupowanych na stany
select distinct a.state, a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a 
group by a.city, a.state 
order by a.state 

--tabela z ilością wypadkow dla poszczególnych miast pogrupowanych na stany z podsumowaniem
select distinct a.state, a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a 
group by cube(a.city, a.state )
order by a.state, ilosc_wypadkow desc

--ilość wszystkich wypadków zarejestrowanych w miastach: 16946
select count (city) from
	(select distinct a.state, a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a 
group by a.city, a.state 
order by ilosc_wypadkow desc ) 

--ilosc miast, w których wydarzył się tylko jeden wypadek wynosi: 1110
select count (*) from
	(select a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a 
group by a.city
having count (a.id) = 1) as foo

----ilosc miast, w których wydarzyły się 1 lub 2 wypadki wynosi: 1929
select count (*) from
	(select city, 
count (a.id) ilosc_wypadkow
from us_accidents a 
group by a.city
having count (a.id) between 1 and 2) as foo

--widok dla wypadków w poszczegolnych miastach
create view v_wypadki_miasta as
select distinct a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a
group by a.city
order by ilosc_wypadkow desc

--ranking miast w odniesienu do ilosci wypadkow
select *, 
dense_rank() over (order by ilosc_wypadkow desc) 
from v_wypadki_miasta  
order by ilosc_wypadkow desc

--podstawowe funkcje statystyczne dla ilosci wypadkow w poszczegolnych miastach (ilosc wszystkich miast 11682)
select
count(city) ilosc_miast,
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker,
percentile_disc(0.95) within group (order by ilosc_wypadkow) kwantyl_95,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow),2) wariancja
from v_wypadki_miasta

--ilosc miast odstających z góry:  1653
select count(*) ilosc_miast_odstajacych from
(
select distinct a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a
group by a.city
having count (a.id)  > 249
order by ilosc_wypadkow desc) foo

-- lista miast odstających z góry:
select distinct a.city, 
count (a.id) ilosc_wypadkow
from us_accidents a
group by a.city
having count (a.id)  > 249
order by ilosc_wypadkow desc

/*
    • rozkład danych (kwantyle, kwartyle, IQR -środkowa połowa zbioru danych):
        ◦ pierwszy kwartyl (Q1) – 25% obserwacji położonych jest poniżej, a 75% powyżej: 4 
        ◦ drugi kwartyl (Q2) (mediana) dzieli zbiór obserwacji na dwie równe części: 20
        ◦ trzeci kwartyl (Q3) – 75% obserwacji położona jest poniżej, a 25% powyżej: 102
        ◦ IQR -środkowa połowa zbioru danych: 98
        ◦ granica obserwacji odstających z góry (upper whisker): 249
        ◦ kwantyl_10: 2
        ◦ kwantyl_90: 394
    • tendencja centralna (średnia, mediana, moda), 
        ◦ średnia: 244
        ◦ moda (dominująca ilość wypadków): 1
   		◦ mediana: 20
    • dyspersja
        ◦ min: 1
        ◦ max: 106 966
        ◦ rozstęp danych: 106 965
        ◦ odchylenie standardowe: 1733


Zbiór obserwacji jest bardzo zróżnicowany. Wysokie wartości odchylenia standardowego świadczą o silnym rozproszeniu wartości wokół średniej.
Do wartości odstających (IQR) można zaliczyć 1656 miast.*/


/*Sprawdzenie tezy: czy odrzucenie miast, w których wydarzył się tylko 1 wypadek (moda) na przestrzeni 5 lat, 
 wpłynie znacząco na statystyki. */

--liczba  miast, w krórych wydarzł się tylko 1 wypadek: 1110 (dominująca ilośc wypadkow wynosiła 1) 
select 
sum (case when vwm.ilosc_wypadkow = 1 then 1 else 0 end) liczba_miast_z_1_wypadkiem
from v_wypadki_miasta vwm 

--liczba_miast_z_1_lub_2_wypadkami: 1929
select 
sum (case when vwm.ilosc_wypadkow between 1 and 2 then 1 else 0 end) liczba_miast_z_1_lub_2_wypadkami
from v_wypadki_miasta vwm 

--podstawowe funkcje statystyczne dla ilosci wypadkow w poszczegolnych miastach, z pominięcim miast z 1 wypadkiem
select 
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow)) wariancja
from v_wypadki_miasta
where ilosc_wypadkow >1

-- porównanie posumowania statystyki dla wypadków we wszystkich miastach vs. miastach z więcej niż 1 wypdkiem

create table wypadki_miasta_wszystkie_vs_miasta_powyżej_1_wypadku
as
select 'wszystkie miasta' as zrodlo_danych,
count(city) ilosc_wszystkich_miast,
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker,
percentile_disc(0.95) within group (order by ilosc_wypadkow) kwantyl_95,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow),2) wariancja
from v_wypadki_miasta

union

select 'miasta powyżej 1 wypadku' as zrodlo_danych,
count(city) ilosc_miast_powyzej_1_wypadku,
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker,
percentile_disc(0.95) within group (order by ilosc_wypadkow) kwantyl_95,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow),2) wariancja
from v_wypadki_miasta
where ilosc_wypadkow >1


select * from wypadki_miasta_wszystkie_vs_miasta_powyżej_1_wypadku

/*Wniosek: odrzucenie miast, w których wydarzł się tylko jeden wypadek (jako najcześciej wystepujący) 
nie wpływa znacząco na posumowanie statystyk dla ilości wypadków w  miastach.*/


/*STATYSTYKI PODSUMOWUJĄCE DOTYCZĄCE LICZBY WYPADKÓW DLA POŁOŻENIA GEOGRAFICZNEGO (zmienne: start_lat, start_lng)*/

--widok na ilość wypadkow w danym miescie/polożeniu geograficznym - poczatek wypadku
create view v_poczatek_wypadku as
select ua.state, ua.city, ua.start_lat, ua.start_lng, count (ua.id) ilosc_wypadkow
from us_accidents ua 
group by ua.state,ua.city,ua.start_lat,ua.start_lng
order by count (ua.id) desc

--ilość wypadków w danym położeniu geograficznym - początek wypadku
select ua.start_lat, ua.start_lng, count (ua.id) ilosc_wypadkow
from us_accidents ua 
group by ua.start_lat,ua.start_lng
order by count (ua.id) desc


--statystyki dla ilosci wypadków w odnienisniu do położenia geograficznego dla v_poczatek_wypadku 
select 
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow)) wariancja
from v_poczatek_wypadku

--ilość miejsc, w których liczba wypadków przybiera wartości odstające (ilosc obserwacji odstjących z gory): 114 991
select count(*) from v_poczatek_wypadku where ilosc_wypadkow >4

--tablea liości miejsc przekraczająca warości średnie
select * from 
(
select '1' as lp, 'miejsca powyzej 100' as anomalie, count(*) from v_poczatek_wypadku where ilosc_wypadkow >100 
union
select '2', 'miejsca powyzej 200', count(*) from v_poczatek_wypadku where ilosc_wypadkow >200
union
select '3', 'miejsca powyzej 300', count(*) from v_poczatek_wypadku where ilosc_wypadkow >300
) foo
order by lp

/*Statystyki osobno dla szerokosci i dla dlugosci geograficznej*/

--ilosc wypadkow na szerokosci geograficznej (poczatek wypadku)
select ua.start_lat szerokoscgeo_poczatek_wypadku, count (ua.id) ilosc_wypadkow
from us_accidents ua 
group by ua.start_lat
order by count (ua.id) desc

--statystyki dla ilości wypadków na badanej szerokosci geograficznej (poczatek wypadku)
select 
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow)) wariancja
from 
	(select ua.start_lat szerokoscgeo_poczatek_wypadku, count (ua.id) ilosc_wypadkow
from us_accidents ua 
group by ua.start_lat
order by count (ua.id) desc) as alias

--ilosc wypadkow na dlugości geograficznej (poczatek wypadku)
select ua.start_lng dlugoscgeo_poczatek_wypadku, count (ua.id) ilosc_wypadkow
from us_accidents ua 
group by ua.start_lng
order by count (ua.id) desc

--statystyki dla ilości wypadków na badanej długości geograficznej (poczatek wypadku)
select 
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow)) wariancja
from 
	(select ua.start_lng dlugoscgeo_poczatek_wypadku, count (ua.id) ilosc_wypadkow
from us_accidents ua 
group by ua.start_lng
order by count (ua.id) desc) as foo


/*

ANALIZA DWUCZYNNIKOWA

Współczynnik korelacji Pearsona - współczynnik korelacji mierzy, jak dobrze dwie zmienne pasują do trendu liniowego. 
Im wynik jest bliższy zeru, tym korelacja jest słabsza. Im wyższa wartość bezwzględna współczynnika korelacji Pearsona, 
tym bardziej prawdopodobne jest, że punkty pasują do linii prostej.*/

			
--KORELACJA POMIĘDZY ILOŚCIĄ WYPADKÓW A POŁOŻENIEM GEOGRAFICZNYM


--korelacja pomiedzy iloscia wypadkow w stanach a szerokoscia geograficzna: - 0,324 (umiarkowana korelacja ujemna)
select round(corr(ilosc_wypadkow,start_lat)::numeric,3) from v_wypadki_stany vws join us_accidents ua on vws.state  = ua.state 

--korelacja pomiedzy iloscia wypadkow w stanach a dlugoscia geograficzna: - 0,657 (bardzo mocna korelacja ujemna)
select round(corr(ilosc_wypadkow,start_lng)::numeric,3) from v_wypadki_stany vws join us_accidents ua on vws.state  = ua.state 

--korelacja pomiedzy iloscia wypadkow w miastach a szerokoscia geograficzna: 0,465 (mocna korelacja ujemna)
select round(corr(ilosc_wypadkow,start_lat)::numeric,3) from v_wypadki_miasta vwm  join us_accidents ua on vwm.city  = ua.city

--korelacja pomiedzy iloscia wypadkow w miastach a dlugoscia geograficzna: -0,094 (słaba lub nieistniejąca korelacja)
select round(corr(ilosc_wypadkow,start_lng)::numeric,3) from v_wypadki_miasta vwm  join us_accidents ua on vwm.city  = ua.city

--korelacja pomiedzy powagą wypadku a szerokoscia geograficzna: 0,089 (słaba lub nieistniejąca korelacja)
select round(corr(severity ,start_lat)::numeric,3) from us_accidents ua 

--korelacja pomiedzy powagą wypadku a dlugoscia geograficzna: 0,114 (słaba lub nieistniejąca korelacja)
select round(corr(severity ,start_lng)::numeric,3) from us_accidents ua 

--korelacja pomiedzy iloscia wypadkow a szerokoscia geograficzna: -0,048  (słaba lub nieistniejąca korelacja)
select corr(start_lat,ilosc_wypadkow) from v_poczatek_wypadku vpw

--korelacja pomiedzy iloscia wypadkow a dlugoscia geograficzna: -0,027 (słaba lub nieistniejąca korelacja)
select corr(start_lng,ilosc_wypadkow) from v_poczatek_wypadku vpw


/*Wniosek: wyniki korelacji badanych zmiennych nie wskazują na związek przyzynowo-skutkowy, jednak
zależność między badanymi zmiennymi może być nieliniowa. np: silna nieliniowa zależność o niskim współczynniku korelacji*/

--KORELACJA POMIĘDZY ILOŚCIĄ WYPADKÓW A ILOŚCIĄ POWAŻNYCH WYPADKÓW (serverity 4) 
--w poszczególnych stanach: 0,971 - bardzo silna korelacja dodatnia

create table wypadki_powaga_vs_ilosc as
select distinct a.state,
count (a.id) ilosc_wypadkow, 
sum (case when severity=4 then 1 else 0 end) ilosc_poważnych_wypadków 
from us_accidents a 
group by cube (a.state)
order by ilosc_wypadkow desc 

select round(corr(ilosc_wypadkow, ilosc_poważnych_wypadków)::numeric,3) as korelacja 
from wypadki_powaga_vs_ilosc 


--KORELACJA POMIĘDZY ILOŚCIĄ WYPADKÓW A ILOŚCIĄ POWAŻNYCH WYPADKÓW (serverity 4) 
--w stanach po odrzuceniu outliers: 0,984 - bardzo silna korelacja dodatnia

-- tabela dla ilosci powaznych wypadków po odrzuceniu stanów odstających
create table wypadki_powaga_vs_ilosc_bez_outliers as
select distinct a.state,
count (a.id) ilosc_wypadkow, 
sum (case when severity=4 then 1 else 0 end) ilosc_poważnych_wypadków 
from us_accidents a 
where a.state != 'California' and a.state != 'TX' and a.state != 'OR' and a.state != 'Floria' 
group by cube (a.state)
order by ilosc_wypadkow desc 

select round(corr(ilosc_wypadkow, ilosc_poważnych_wypadków)::numeric,3) as korelacja 
from wypadki_powaga_vs_ilosc_bez_outliers


/*Wniosek: wyniki korelacji badanych zmiennych wskazują na silną zależność liniową pomiędzy ilością wszystkich wypadków i powaznych wypadków
zależność między badanymi zmiennymi może być nieliniowa. np: silna nieliniowa zależność o niskim współczynniku korelacji*/

/*ANALIZA DLA ZMIENNEJ 'serverity' OKREŚLAJĄCEJ POWAGĘ WYPADKU*/

--ilosc powaznych wypadkow: 131 193  (maksymalna powaga wypadku - 4)
select 
sum (case when severity=4 then 1 else 0 end) ilosc_poważnych_wypadków
from us_accidents ua 

--ilosc powaznych wypadkow w poszczegolnych stanach
select distinct ua.state ,
sum (case when severity=4 then 1 else 0 end) over (partition by ua.state) ilosc_poważnych_wypadków 
from us_accidents ua 
order by ilosc_poważnych_wypadków desc

--tabela dla ilości wypadkow w poszczegolnych stanach vs. ilosc poważnych wypadkow 
select distinct a.state,
count (a.id) ilosc_wypadkow, 
sum (case when severity=4 then 1 else 0 end) ilosc_poważnych_wypadków 
from us_accidents a 
group by cube (a.state)
order by ilosc_wypadkow desc 

--widok na ilość wypadkow i ich powage w danym miescie
create view v_wypadki_miasta_powaga as
select distinct ua.state, ua.city, ua.severity,
count (ua.id) ilosc_wypadkow
from us_accidents ua 
group by ua.state, ua.city,ua.severity 
order by ua.severity desc, ilosc_wypadkow desc

--ilosc poważnych wypadkow w poszczegolnych poszczegolnych miastach
select distinct ua.city ,
sum (case when severity=4 then 1 else 0 end) over (partition by ua.city) ilosc_poważnych_wypadków 
from us_accidents ua 
order by ilosc_poważnych_wypadków desc


--porównanie ilości wypadków na terenie zabudowanym i niezabudowanym (drogi międzymiastowe, drogi międzystanowe)
--stworzenie tabeli dla ulic z numerem domu - tzałożenie dla wyznaczenia terenu zabudowanego

create table numery_miasta_stany as
select  ua.id , ua.state, ua.city , ua.street , ua."Number" , LENGTH(ua."Number") teren_zabudowany
from us_accidents ua 
where LENGTH(ua."Number")>0

select * from numery_miasta_stany 

select distinct state, city, 
count (id) ilosc_wypadkow
from numery_miasta_stany 
group by city, state 
order by ilosc_wypadkow desc

--podstawowe funkcje statystyczne dla ilosci wypadkow w miastach na terenie zabudowanym
select
count(city) ilosc_miast,
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
percentile_disc(0.1) within group (order by ilosc_wypadkow) kwantyl_10,
percentile_disc(0.9) within group (order by ilosc_wypadkow) kwantyl_90,
percentile_disc(0.25) within group (order by ilosc_wypadkow) kwantyl_25,
percentile_disc(0.75) within group (order by ilosc_wypadkow) kwantyl_75,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker,
percentile_disc(0.95) within group (order by ilosc_wypadkow) kwantyl_95,
(max(ilosc_wypadkow) - min(ilosc_wypadkow)) as rozstep_danych,
round(stddev(ilosc_wypadkow)) odchylenie_standardowe,
round(variance (ilosc_wypadkow),2) wariancja
from
(
select distinct state, city, 
count (id) ilosc_wypadkow
from numery_miasta_stany 
group by city, state 
order by ilosc_wypadkow desc
) foo

--porównanie podstawowych funkcji statystycznych dla ilosci wypadkow w miastach na terenie zabudowanym i całym obszarze
select 'teren zabudowany' as zrodlo,
count(city) ilosc_miast,
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker
from
	(
		select distinct state, city, 
		count (id) ilosc_wypadkow
		from numery_miasta_stany 
		group by city, state 
		order by ilosc_wypadkow desc
	) foo
union 
select 'cały obszar' as zrodlo,
count(city) ilosc_miast,
min(ilosc_wypadkow),
max(ilosc_wypadkow),
round(avg (ilosc_wypadkow)) srednia_ilosc_wypadkow,
mode() within group (order by ilosc_wypadkow) moda_domiujaca_ilosc_wypadkow,
percentile_disc(0.5) within group (order by ilosc_wypadkow) mediana,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) as IQR,
(percentile_disc(0.75) within group (order by ilosc_wypadkow) 
+ 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as upper_whisker,
(percentile_disc(0.25) within group (order by ilosc_wypadkow) 
- 1.5 *(percentile_disc(0.75) within group (order by ilosc_wypadkow) - percentile_disc(0.25) within group (order by ilosc_wypadkow)) ) 
as lower_whisker
from v_wypadki_miasta vwm 


/*PODSUMOWANIE

1. ilość wypadków drogowych w USA zależy do położenia geograficznego, 

2. pomimo niskiego współczynnika korelacji:

	  * wstepuje ponad 170 punków współrzędnych geograficnzych, 
	  w których ilość wypadków drogowych ponad 100-krotnie przekracza wartości średnie
	  
	  * wstepuje ponad 10 punków współrzędnych geograficnzych, 
	  w których ilość wypadków drogowych ponad 200-krotnie przekracza wartości średnie
	  
	  * w 1 charekterystycznym punkcie ilość wypadków drogowych ponad 300-krotnie przekracza wartości średnie
	  
3. obszary, w których najczęściej dochodzi do wypadków drogowych obejmują cztery stany:
   California, Floryda, Texas, Oregon:
	   		(a) California - ilość wypadków 14-krotnie przekracza średnią a ponad 4000 razy dominującą ilość wypadków
	   		(b) Floryda - ilość wypadków 7-krotnie przekracza średnią a ponad 2000 razy dominującą ilość wypadków
	   		(c) Texas - ilość wypadków 3-krotnie przekracza średnią a ponad 700 razy dominującą ilość wypadków
	   		(d) Oregon - ilość wypadków 2-krotnie przekracza średnią a ponad 600 razy dominującą ilość wypadków

3. top rankingu miast,w ktorych dochodzi do największej liczby wypadków:
			(a) Miami - ilość wypadków 400 razy przekracza średnią, w tym 65 % na terenie zabudowanym
			(b) Los Angeles  - ilość wypadków 280 azy przekracza średnią, w tym ponad 30% na terenie zabudowanym
			(c) Orlando - ilość wypadków 220 azy przekracza średnią, w tym ponad 70% na terenie zabudowanym.
						
 */




