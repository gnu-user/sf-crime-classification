-- Extensions for geolocation
create extension cube;
create extension earthdistance;

CREATE SCHEMA IF NOT EXISTS staging;

-- Create staging tables
CREATE TABLE staging.incidents
(
    incident_id     TEXT,
    category        TEXT,
    description     TEXT,
    day_of_week     TEXT,
    date            TEXT,
    time            TEXT,
    district        TEXT,
    resolution      TEXT,
    address         TEXT,
    x               TEXT,
    y               TEXT,
    location        TEXT,
    pd_id           TEXT
);

CREATE TABLE staging.train
(
    date            TEXT,
    category        TEXT,
    description     TEXT,
    day_of_week     TEXT,
    district        TEXT,
    resolution      TEXT,
    address         TEXT,
    x               TEXT,
    y               TEXT
);

CREATE TABLE staging.test
(
    incident_id     TEXT,
    date            TEXT,
    day_of_week     TEXT,
    district        TEXT,
    address         TEXT,
    x               TEXT,
    y               TEXT
);

CREATE TABLE staging.answers
(
    incident_id     TEXT,
    date            TEXT,
    category        TEXT,
    day_of_week     TEXT,
    district        TEXT,
    address         TEXT,
    x               TEXT,
    y               TEXT
);

-- Load in the staging data
\copy staging.incidents from '../data/sf_crime.csv' CSV HEADER;
\copy staging.train from '../data/train.csv' CSV HEADER;
\copy staging.test from '../data/test.csv' CSV HEADER;
\copy staging.answers from '../data/answers.csv' CSV HEADER;


CREATE or REPLACE FUNCTION percent_unique(table_name REGCLASS, column_name TEXT,
                                          OUT result NUMERIC)
LANGUAGE PLPGSQL
AS
$$
BEGIN
  EXECUTE FORMAT('
    WITH uni   AS (SELECT COUNT(*) FROM (SELECT distinct %I FROM %s) t),
         total AS (SELECT count(*) FROM %s)
    SELECT round((100::numeric * uni.count) / total.count, 2)
    FROM uni, total', column_name, table_name, table_name)
  INTO result;
END
$$;


CREATE FUNCTION normalize_column(source_table REGCLASS, source_column TEXT,
                                 target_table TEXT,
                                 target_key_column TEXT,
                                 target_value_column TEXT,
                                 target_value_type TEXT)
RETURNS TEXT
LANGUAGE PLPGSQL
AS
$$
BEGIN
  EXECUTE FORMAT('
    CREATE TABLE %s
    (
        %I SERIAL NOT NULL PRIMARY KEY,
        %I %s NOT NULL
    )', target_table, target_key_column,
        target_value_column, target_value_type);
  EXECUTE FORMAT('
    INSERT INTO %s (%I) SELECT DISTINCT %I::%s FROM %s
    ', target_table, target_value_column,
       source_column, target_value_type,
       source_table);
  RETURN target_table;
END
$$;

select normalize_column('staging.incidents', 'address', 'addresses', 'address_id', 'address', 'TEXT');
select normalize_column('staging.incidents', 'resolution', 'resolutions', 'resolution_id', 'resolution', 'TEXT');
select normalize_column('staging.incidents', 'description', 'descriptions', 'description_id', 'description', 'TEXT');
select normalize_column('staging.incidents', 'category', 'categories', 'category_id', 'category', 'TEXT');
select normalize_column('staging.incidents', 'district', 'districts', 'district_id', 'district', 'TEXT');

CREATE TABLE incidents
(
    incident_id     SERIAL    NOT NULL PRIMARY KEY,
    incident_num    BIGINT    NOT NULL,
    incident_time   TIMESTAMP NOT NULL,
    category_id     INTEGER   NOT NULL REFERENCES categories   (category_id),
    description_id  INTEGER   NOT NULL REFERENCES descriptions (description_id),
    district_id     INTEGER   NOT NULL REFERENCES districts    (district_id),
    resolution_id   INTEGER   NOT NULL REFERENCES resolutions  (resolution_id),
    address_id      INTEGER   NOT NULL REFERENCES addresses    (address_id),
    location        POINT     NOT NULL,
    pd_id           BIGINT    NOT NULL
);

CREATE TABLE train
(
    incident_id     SERIAL    NOT NULL PRIMARY KEY,
    incident_time   TIMESTAMP NOT NULL,
    category_id     INTEGER   NOT NULL REFERENCES categories   (category_id),
    description_id  INTEGER   NOT NULL REFERENCES descriptions (description_id),
    district_id     INTEGER   NOT NULL REFERENCES districts    (district_id),
    resolution_id   INTEGER   NOT NULL REFERENCES resolutions  (resolution_id),
    address_id      INTEGER   NOT NULL REFERENCES addresses    (address_id),
    location        POINT     NOT NULL,
    pd_id           BIGINT    NOT NULL
);

    date            TEXT,
    category        TEXT,
    description     TEXT,
    day_of_week     TEXT,
    district        TEXT,
    resolution      TEXT,
    address         TEXT,
    x               TEXT,
    y               TEXT


-- LOAD DATA
BEGIN;
INSERT INTO incidents (incident_num, incident_time, category_id, description_id, district_id, resolution_id, address_id, location, pd_id)
SELECT incident_id::BIGINT,
       date::TIMESTAMP + time::INTERVAL,
       category_id,
       description_id,
       district_id,
       resolution_id,
       address_id,
       POINT(y::NUMERIC, x::NUMERIC),
       pd_id::BIGINT
FROM staging.incidents
NATURAL JOIN addresses
NATURAL JOIN resolutions
NATURAL JOIN descriptions
NATURAL JOIN categories
NATURAL JOIN districts;
END;

-- DELETE STAGING DATA
drop table staging.incidents;