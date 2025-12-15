-- -- Création de l'utilisateur (si n'existe pas)
-- DO
-- $do$
-- BEGIN
--    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'churn_user') THEN
--       CREATE USER churn_user WITH PASSWORD 'Jtm123mama';
--    END IF;
-- END
-- $do$;

-- -- Création de la base de données
-- DO
-- $do$
-- BEGIN
--    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'churn_db') THEN
--       CREATE DATABASE churn_db;
--    END IF;
-- END
-- $do$;

-- -- Connexion à la base (à exécuter dans psql)
-- \connect churn_db;

-- -- Création de la table predictions
-- CREATE TABLE IF NOT EXISTS predictions (
--     id SERIAL PRIMARY KEY,
--     customer_id TEXT,
--     prediction FLOAT,
--     prob FLOAT,
--     ts TIMESTAMP DEFAULT now()
-- );
-- -- Attribution des droits à l'utilisateur*
-- GRANT ALL PRIVILEGES ON TABLE predictions TO churn_user;
-- GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_user;
-- -- *

-- -- Création de la base de données (si n'existe pas) (à exécuter dans psql)
-- -- Création de l'utilisateur (si n'existe pas) (à exécuter dans psql)
DO 
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'churn_user') THEN
      CREATE USER churn_user WITH PASSWORD 'Jtm123mama';
   END IF;
END
$do$;

