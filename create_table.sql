-- Execute this command to create the tables
-- sqlite3 olist.db < create_table.sql

-- Create a table for Customers
CREATE TABLE IF NOT EXISTS Customers (
    customer_id INTEGER PRIMARY KEY,
    customer_unique_id TEXT,
    customer_zip_code_prefix TEXT,
    customer_city TEXT ,
    customer_state TEXT
);

-- Create a table for Geolocalisation
CREATE TABLE IF NOT EXISTS Geolocalisation (
    geolocation_zip_code_prefix TEXT,
    geolocation_lat REAL,
    geolocation_lng REAL,
    geolocation_city TEXT,
    geolocation_state TEXT
);

