/*liczenie wierszy*/
select count(*) from us_accidents ua 


/* wczytanie całej tabeli*/
select *
from us_accidents ua 

/*wczytanie potrzebnych kolumn*/
select id ,severity ,amenity ,bump ,crossing ,give_way ,junction ,no_exit ,railway ,roundabout ,station ,stop ,traffic_calming ,traffic_signal ,turning_loop 
from us_accidents ua 

/*tworzenie widoku*/
create view v_POI as
select id ,severity ,amenity ,bump ,crossing ,give_way ,junction ,no_exit ,railway ,roundabout ,station ,stop ,traffic_calming ,traffic_signal ,turning_loop 
from us_accidents ua 

/*sprawdzenie*/
select *
from v_poi vp 

/*liczenie*/
select  distinct amenity ,
count(id) over (partition by amenity = 'True') amen_count
from v_poi vp 

/*liczenie wartości TRUE*/
SELECT 
COALESCE(sum(CASE WHEN amenity  THEN 1 ELSE 0 END),0)
from v_poi vp 

/*liczenie wartości wszystkich kolumn*/
select  distinct amenity = 'True', bump = 'True',
count(id) over (partition by amenity = 'True') amen_count
from v_poi vp 


/*liczenie wartości TRUE w poszczególnych kolumnach*/
select count(id), 
count(*) filter (where amenity) amenity_number,
count(*) filter (where bump) bump_number,
count(*) filter (where crossing) crossing_number,
count(*) filter (where give_way) give_number,
count(*) filter (where junction) junction_number,
count(*) filter (where no_exit) exit_number,
count(*) filter (where railway) railway_number,
count(*) filter (where roundabout) round_number,
count(*) filter (where station) station_number,
count(*) filter (where stop) stop_number,
count(*) filter (where traffic_calming) calming_number,
count(*) filter (where traffic_signal) signal_number,
count(*) filter (where turning_loop) loop_number
from v_poi vp 


/*tworzenie tabeli z liczbami*/
SELECT count(id), count(*) filter (where amenity), 
       ROUND(amenity  * 100.0 / id, 1) AS Percent
FROM v_poi vp 

/*ilość stanów w usa*/
select distinct state 
from us_accidents ua 


/* wartości TRUE z podziałem na stany*/
select state ,
count(*) filter (where amenity) amenity_number,
count(*) filter (where bump) bump_number,
count(*) filter (where crossing) crossing_number,
count(*) filter (where give_way) give_number,
count(*) filter (where junction) junction_number,
count(*) filter (where no_exit) exit_number,
count(*) filter (where railway) railway_number,
count(*) filter (where roundabout) round_number,
count(*) filter (where station) station_number,
count(*) filter (where stop) stop_number,
count(*) filter (where traffic_calming) calming_number,
count(*) filter (where traffic_signal) signal_number,
count(*) filter (where turning_loop) loop_number
from us_accidents ua 
group by state 


/*wartości false, true i suma z podziałem na stany*/
select state ,
	count(nullif (amenity = false, true )) _true,
	count(nullif (amenity = true , true )) _false,
	count(amenity) _sum
from us_accidents ua 
group by state


/* wartości true i suma z podziałem na stany*/
select state ,
	count(nullif (amenity = false, true )) amenity_true,
	count(nullif (bump = false , true )) bump_true,
	count(nullif (crossing = false , true )) crossing_true,
	count(nullif (give_way = false , true )) give_sum,
	count(nullif (junction = false , true )) junction_sum,
	count(nullif (no_exit = false , true )) exit_sum,
	count(nullif (railway = false , true )) railway_sum,
	count(nullif (roundabout = false , true )) round_sum,
	count(nullif (station = false , true )) station_sum,
	count(nullif (stop = false , true )) stop_sum,
	count(nullif (traffic_calming = false , true ))calming_sum,
	count(nullif (traffic_signal = false , true ))signal_sum,
	count(nullif (turning_loop = false , true ))loop_sum,
	count(amenity) POI_sum
from us_accidents ua 
group by state


/*stworzenei widoku v2*/
create view v_poi2 as
select state ,
	count(nullif (amenity = false, true )) amenity_true,
	count(nullif (bump = false , true )) bump_true,
	count(nullif (crossing = false , true )) crossing_true,
	count(nullif (give_way = false , true )) give_true,
	count(nullif (junction = false , true )) junction_true,
	count(nullif (no_exit = false , true )) exit_true,
	count(nullif (railway = false , true )) railway_true,
	count(nullif (roundabout = false , true )) round_true,
	count(nullif (station = false , true )) station_true,
	count(nullif (stop = false , true )) stop_true,
	count(nullif (traffic_calming = false , true ))calming_true,
	count(nullif (traffic_signal = false , true ))signal_true,
	count(nullif (turning_loop = false , true ))loop_true,
	count(amenity) POI_sum
from us_accidents ua 
group by state



/*sprawdzenie widoku*/
select *
from v_poi2 vp 



/*liczenie procentów*/
select *,
	(amenity_true ::float / poi_sum ::float) * 100 as amenity_per,
	(bump_true ::float / poi_sum::float) * 100 as bum_per,
	(crossing_true ::float / poi_sum ::float) * 100 as crossing_per,
	(give_true ::float / poi_sum ::float) * 100 as give_per,
	(junction_true ::float / poi_sum ::float) * 100 as junction_per,
	(exit_true ::float / poi_sum ::float) * 100 as exit_per,
	(railway_true ::float / poi_sum ::float) * 100 as railway_per,
	(round_true ::float / poi_sum ::float) * 100 as round_per,
	(station_true ::float / poi_sum ::float) * 100 as station_per,
	(stop_true ::float / poi_sum ::float) * 100 as stop_per,
	(calming_true ::float / poi_sum ::float) * 100 as calming_per,
	(signal_true ::float / poi_sum ::float) * 100 as signal_per,
	(loop_true ::float / poi_sum ::float) * 100 as loop_per
from v_poi2 vp 


/*stworzenie widoku z %*/
create view v_poi3 as
select *,
	(amenity_true ::float / poi_sum ::float) * 100 as amenity_per,
	(bump_true ::float / poi_sum::float) * 100 as bum_per,
	(crossing_true ::float / poi_sum ::float) * 100 as crossing_per,
	(give_true ::float / poi_sum ::float) * 100 as give_per,
	(junction_true ::float / poi_sum ::float) * 100 as junction_per,
	(exit_true ::float / poi_sum ::float) * 100 as exit_per,
	(railway_true ::float / poi_sum ::float) * 100 as railway_per,
	(round_true ::float / poi_sum ::float) * 100 as round_per,
	(station_true ::float / poi_sum ::float) * 100 as station_per,
	(stop_true ::float / poi_sum ::float) * 100 as stop_per,
	(calming_true ::float / poi_sum ::float) * 100 as calming_per,
	(signal_true ::float / poi_sum ::float) * 100 as signal_per,
	(loop_true ::float / poi_sum ::float) * 100 as loop_per
from v_poi2 vp 

select *
from v_poi3 vp 

/* zaokrąglenie % do 2 wartości po przecinku*/
select state ,
	round(amenity_per ::numeric ,2) amenity_per ,
	round(bum_per ::numeric ,2) bump_per,
	round(crossing_per ::numeric ,2) crossing_per ,
	round(give_per ::numeric ,2) give_per ,
	round(junction_per ::numeric ,2) junction_per ,
	round(exit_per ::numeric ,2) exit_per,
	round(railway_per ::numeric ,2) railway_per ,
	round(round_per ::numeric ,2) round_per ,
	round(station_per ::numeric ,2) station_per ,
	round(stop_per ::numeric ,2) stop_per ,
	round(calming_per ::numeric ,2) calming_per ,
	round(signal_per ::numeric ,2) signal_per ,
	round(loop_per ::numeric ,2) loop_per 
from v_poi3 vp 

select *
from v_poi3 vp 


select state , avg(severity)
from us_accidents ua 
group by state 

/* widok zaokrąglony z % */
create view v_poi4 as
select state ,
	round(amenity_per ::numeric ,2) amenity_per ,
	round(bum_per ::numeric ,2) bump_per,
	round(crossing_per ::numeric ,2) crossing_per ,
	round(give_per ::numeric ,2) give_per ,
	round(junction_per ::numeric ,2) junction_per ,
	round(exit_per ::numeric ,2) exit_per,
	round(railway_per ::numeric ,2) railway_per ,
	round(round_per ::numeric ,2) round_per ,
	round(station_per ::numeric ,2) station_per ,
	round(stop_per ::numeric ,2) stop_per ,
	round(calming_per ::numeric ,2) calming_per ,
	round(signal_per ::numeric ,2) signal_per ,
	round(loop_per ::numeric ,2) loop_per 
from v_poi3 vp

select *
from v_poi4 vp 
