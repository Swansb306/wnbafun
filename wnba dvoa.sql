ALTER TABLE teamelo
ADD COLUMN id SERIAL

---trying to get average scores against each team
SELECT team1, team2, season, avg(score1) AS aver1, avg(score2) AS aver2
FROM teamelo
GROUP BY season, team1, team2
---try something else
WITH t1 AS (
SELECT team1, season, avg(score2) AS avgagainstteam1
FROM teamelo 
GROUP BY season, team1
),
t2 AS (
SELECT team2, season, avg(score1) AS avgagainstteam2
FROM teamelo
GROUP BY season, team2
)
SELECT teamelo.team1, teamelo.team2, teamelo.season, t1.avgagainstteam1, t2.avgagainstteam2
FROM teamelo
INNER JOIN t1
ON teamelo.team1 = t1.team1
AND teamelo.season = t1.season
INNER JOIN t2
ON teamelo.team2 = t2.team2
AND teamelo.season = t2.season
--okay satisfied that that gave you average against
--now we subtract score from avg against 
--to get some kind of score
CREATE TABLE averageagainst AS
WITH t1 AS (
SELECT team1, season, avg(score2) AS avgagainstteam1
FROM teamelo 
GROUP BY season, team1
),
t2 AS (
SELECT team2, season, avg(score1) AS avgagainstteam2
FROM teamelo
GROUP BY season, team2
)
SELECT teamelo.team1, teamelo.team2, teamelo.season, t1.avgagainstteam1, t2.avgagainstteam2
FROM teamelo
INNER JOIN t1
ON teamelo.team1 = t1.team1
AND teamelo.season = t1.season
INNER JOIN t2
ON teamelo.team2 = t2.team2
AND teamelo.season = t2.season


SELECT t1.team1, t1.team2, t1.score1, t1.score2, 
t2.avgagainstteam1, t2.avgagainstteam2, t1.season, t1.score1 - t2.avgagainstteam2 AS off1, t1.score2 - t2.avgagainstteam1 AS off2
FROM teamelo t1
INNER JOIN averageagainst t2
ON t1.team1 = t2.team1
AND t1.season = t2.season
AND t1.team2 = t2.team2

--what if we aggregated by season? 
WITH part1 AS (
SELECT t1.team1, t1.team2, t1.score1, t1.score2, 
t2.avgagainstteam1, t2.avgagainstteam2, t1.season, 
t1.score1 - t2.avgagainstteam2 AS off1, 
t1.score2 - t2.avgagainstteam1 AS off2
FROM teamelo t1
INNER JOIN averageagainst t2
ON t1.team1 = t2.team1
AND t1.season = t2.season
AND t1.team2 = t2.team2)
SELECT t.team1, t.team2, t.score1, t.score2, t.season, AVG(p.off1) avgoff1, 
AVG(p.off2) AS avgoff2
FROM teamelo t
INNER JOIN part1 p
ON t.team1 = p.team1
AND t.season = p.season
AND t.team2 = p.team2
GROUP BY t.season, t.team1, t.team2, t.score1, t.score2
ORDER BY t.season, t.team1

--that did not aggregate by season. Try again. 
SELECT t1.team1, t1.team2, t1.score1, t1.score2, 
t2.avgagainstteam1, t2.avgagainstteam2, t1.season, 
avg(t1.score1 - t2.avgagainstteam2) AS off1, 
avg(t1.score2 - t2.avgagainstteam1) AS off2
FROM teamelo t1
INNER JOIN averageagainst t2
ON t1.team1 = t2.team1
AND t1.season = t2.season
AND t1.team2 = t2.team2
GROUP BY t1.season, t1.team1, t1.team2, t1.score1, t1.score2, t2.avgagainstteam1, t2.avgagainstteam2
ORDER BY t1.season, t1.team1
--also not quite right

WITH part1 AS (
SELECT t1.team1, t1.team2, t1.score1, t1.score2, 
t2.avgagainstteam1, t2.avgagainstteam2, t1.season, 
t1.score1 - t2.avgagainstteam2 AS off1, 
t1.score2 - t2.avgagainstteam1 AS off2
FROM teamelo t1
INNER JOIN averageagainst t2
ON t1.team1 = t2.team1
AND t1.season = t2.season
AND t1.team2 = t2.team2),
part2 AS(
SELECT e.team1, e.season, avg(e.score1 - a.avgagainstteam2) AS avgoff1
FROM teamelo e
INNER JOIN averageagainst a
ON e.team1 = a.team1
AND e.season = a.season
GROUP BY e.season, e.team1
),
part3 AS (
SELECT e.team2, e.season,
avg(e.score2 - a.avgagainstteam1) AS avgoff2
FROM teamelo e
INNER JOIN averageagainst a
ON e.team2 = a.team2
AND e.season = a.season
GROUP BY e.season, e.team2
)
SELECT t.team1, t.team2, t.score1, t.score2, t.season, p.off1, p.off2, p2.avgoff1, 
p3.avgoff2
FROM teamelo t
INNER JOIN part1 p
ON t.team1 = p.team1
AND t.season = p.season
AND t.team2 = p.team2
INNER JOIN part2 p2
ON t.team1 = p2.team1
AND t.season = p2.season
INNER JOIN part3 p3
ON t.team2 = p3.team2
AND t.season = p3.season
ORDER BY t.season, t.team1

--correction
--CREATE TABLE pointsagainst AS
WITH t1 AS (
  SELECT team1 AS team, season, avg(score2) AS avgagainst
  FROM teamelo 
  GROUP BY team1, season
),
t2 AS (
  SELECT team2 AS team, season, avg(score1) AS avgagainst
  FROM teamelo 
  GROUP BY team2, season
),
combined_avg AS (
  SELECT team, season, avg(avgagainst) AS avg_points_against
  FROM (
    SELECT * FROM t1
    UNION ALL
    SELECT * FROM t2
  )
  GROUP BY season, team
),
team1_offense AS (
  SELECT teamelo.team1 AS team, teamelo.season,
         AVG(teamelo.score1 - a2.avg_points_against) AS team1_offense_metric
  FROM teamelo
  LEFT JOIN combined_avg a2
    ON teamelo.team2 = a2.team
   AND teamelo.season = a2.season
  GROUP BY teamelo.team1, teamelo.season
),
team2_offense AS (
  SELECT teamelo.team2 AS team, teamelo.season,
         AVG(teamelo.score2 - a1.avg_points_against) AS team2_offense_metric
  FROM teamelo
  LEFT JOIN combined_avg a1
    ON teamelo.team1 = a1.team
   AND teamelo.season = a1.season
  GROUP BY teamelo.team2, teamelo.season
)
SELECT te.id, te.team1, te.team2, te.season,
       a1.avg_points_against AS team1_points_against,
       a2.avg_points_against AS team2_points_against,
       t1o.team1_offense_metric,
       t2o.team2_offense_metric,
       te.date, te.score1, te.score2,
	   te.prob1,te.elo1_pre,te.elo2_pre
FROM teamelo te
LEFT JOIN combined_avg a1 ON te.team1 = a1.team AND te.season = a1.season
LEFT JOIN combined_avg a2 ON te.team2 = a2.team AND te.season = a2.season
LEFT JOIN team1_offense t1o ON te.team1 = t1o.team AND te.season = t1o.season
LEFT JOIN team2_offense t2o ON te.team2 = t2o.team AND te.season = t2o.season

--