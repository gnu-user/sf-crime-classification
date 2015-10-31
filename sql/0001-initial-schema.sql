CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE staging.crime
(
    incident_id TEXT,
    category TEXT,
    description TEXT,
    day_of_week TEXT,
    date TEXT,
    time TEXT,
    district TEXT,
    resolution TEXT,
    address TEXT,
    x TEXT,
    y TEXT,
    location TEXT,
    pd_id TEXT
);

\copy staging.crime from 'sf_crime.csv' CSV HEADER;

CREATE or replace FUNCTION percent_unique(table_name REGCLASS, column_name TEXT,
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

select normalize_column('staging.crime', 'address', 'addresses', 'address_id', 'address', 'TEXT');
select normalize_column('staging.crime', 'resolution', 'resolutions', 'resolution_id', 'resolution', 'TEXT');
select normalize_column('staging.crime', 'description', 'descriptions', 'description_id', 'description', 'TEXT');
select normalize_column('staging.crime', 'category', 'categories', 'category_id', 'category', 'TEXT');
select normalize_column('staging.crime', 'district', 'districts', 'district_id', 'district', 'TEXT');

create extension cube;
create extension earthdistance;

CREATE TABLE crime
(
    incident_id     SERIAL    NOT NULL PRIMARY KEY,
    incident_number BIGINT    NOT NULL,
    incident_time   TIMESTAMP NOT NULL,
    category_id     INTEGER   NOT NULL REFERENCES categories   (category_id),
    description_id  INTEGER   NOT NULL REFERENCES descriptions (description_id),
    district_id     INTEGER   NOT NULL REFERENCES districts    (district_id),
    resolution_id   INTEGER   NOT NULL REFERENCES resolutions  (resolution_id),
    address_id      INTEGER   NOT NULL REFERENCES addresses    (address_id),
    location        POINT     NOT NULL,
    pd_id           BIGINT    NOT NULL
);

INSERT INTO crime (incident_number, incident_time, category_id, description_id, district_id, resolution_id, address_id, location, pd_id)
SELECT incident_id::bigint,
       date::timestamp + time::interval,
       category_id,
       description_id,
       district_id,
       resolution_id,
       address_id,
       POINT(y::numeric, x::numeric),
       pd_id::bigint
FROM staging.crime
NATURAL JOIN addresses
NATURAL JOIN resolutions
NATURAL JOIN descriptions
NATURAL JOIN categories
NATURAL JOIN districts;
