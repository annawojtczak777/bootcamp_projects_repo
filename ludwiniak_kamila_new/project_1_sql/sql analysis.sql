-- start time and end time of accidents date descending -- 
SELECT DISTINCT Start_Time, End_Time
from US_Accidents_Dec21_updated uadu 
ORDER BY start_time DESC,
End_Time DESC 

-- all data --
select * 
from us_accidents_dec21_updated

-- number of accidents / all years --

SELECT
DATE(Start_Time) as dzien,
count(*) as ilosc
from US_Accidents_Dec21_updated
WHERE start_time BETWEEN "2016-01-01" AND "2021-12-31"
GROUP BY dzien
ORDER BY ilosc desc

-- number of accidents in the morning ( 5AM - 12 AM ) --

SELECT
DATE(Start_Time) as dzien,
count(*) as ilosc
from US_Accidents_Dec21_updated
WHERE start_time BETWEEN "2016-01-01 05:00:00" AND "2021-12-31 11:59:59"
GROUP BY dzien
ORDER BY ilosc DESC


-- state with the highest number of accidents (5AM - 12 AM) -- 

SELECT
DATE(Start_Time) as dzien,
count(*) as ilosc
State as stan_USA
from US_Accidents_Dec21_updated
WHERE start_time BETWEEN "2016-01-01 05:00:00" AND "2021-12-31 11:59:59"
GROUP BY stan_USA
ORDER BY ilosc DESC


--  number of accidentsin the afternoon (12am - 6 pm) per state --
SELECT 
DATE(Start_Time) as dzien,
count(*) as ilosc_wypadkow,
State as stan_USA
from US_Accidents_Dec21_updated
WHERE start_time BETWEEN "2016-01-01 12:00:00" AND "2021-12-31 17:59:59"
GROUP BY stan_USA 
ORDER BY ilosc DESC

-- the highest number of accidents at night (6pm - 5am) per state -- 

SELECT
DATE(Start_Time) as dzien,
count(*) as ilosc_wypadkow,
State as stan_USA
from US_Accidents_Dec21_updated
WHERE start_time BETWEEN "2016-01-01 18:00:00" AND "2021-12-31 04:59:59"
GROUP BY stan_USA
ORDER BY  ilosc_wypadkow DESC