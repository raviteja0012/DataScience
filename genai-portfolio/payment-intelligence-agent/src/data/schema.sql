-- Payment Intelligence Agent - Snowflake Schema DDL
-- Database: PAYMENT_DB / Schema: PAYMENT_SCHEMA
--
-- This DDL defines the core payment processing data model supporting
-- transaction analytics, settlement tracking, chargeback management,
-- and fraud detection workflows.

-- ============================================================
-- Database and Schema Setup
-- ============================================================

CREATE DATABASE IF NOT EXISTS PAYMENT_DB;
CREATE SCHEMA IF NOT EXISTS PAYMENT_DB.PAYMENT_SCHEMA;

USE DATABASE PAYMENT_DB;
USE SCHEMA PAYMENT_SCHEMA;

-- ============================================================
-- MERCHANTS - Merchant profiles and risk categorization
-- ============================================================

CREATE TABLE IF NOT EXISTS MERCHANTS (
    MERCHANT_ID         VARCHAR(36)     NOT NULL,
    MERCHANT_NAME       VARCHAR(200)    NOT NULL,
    MCC_CODE            VARCHAR(4)      NOT NULL      COMMENT 'Merchant Category Code (ISO 18245)',
    MCC_DESCRIPTION     VARCHAR(100),
    REGION              VARCHAR(50),
    COUNTRY_CODE        VARCHAR(2)                    COMMENT 'ISO 3166-1 alpha-2',
    RISK_TIER           VARCHAR(10)                   COMMENT 'LOW, MEDIUM, HIGH',
    ONBOARDING_DATE     DATE,
    STATUS              VARCHAR(20)     NOT NULL      COMMENT 'ACTIVE, SUSPENDED, CLOSED',
    MONTHLY_VOLUME_LIMIT DECIMAL(14,2)                COMMENT 'Monthly processing limit in USD',
    CREATED_AT          TIMESTAMP_NTZ   NOT NULL      DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_MERCHANTS PRIMARY KEY (MERCHANT_ID)
);

-- ============================================================
-- CUSTOMERS - Customer profiles (PII-minimized)
-- ============================================================

CREATE TABLE IF NOT EXISTS CUSTOMERS (
    CUSTOMER_ID               VARCHAR(36)     NOT NULL,
    CUSTOMER_HASH             VARCHAR(64)     NOT NULL  COMMENT 'SHA-256 of customer PII for dedup',
    CUSTOMER_SEGMENT          VARCHAR(20)               COMMENT 'PREMIUM, STANDARD, NEW',
    COUNTRY_CODE              VARCHAR(2),
    REGION                    VARCHAR(50),
    ACCOUNT_CREATED_DATE      DATE,
    RISK_SCORE                DECIMAL(5,2)              COMMENT 'Customer risk score 0-100',
    LIFETIME_TRANSACTION_COUNT INTEGER,
    LIFETIME_TRANSACTION_VALUE DECIMAL(14,2),
    CREATED_AT                TIMESTAMP_NTZ   NOT NULL  DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_CUSTOMERS PRIMARY KEY (CUSTOMER_ID)
);

-- ============================================================
-- TRANSACTIONS - Core payment transaction records
-- ============================================================

CREATE TABLE IF NOT EXISTS TRANSACTIONS (
    TRANSACTION_ID      VARCHAR(36)     NOT NULL,
    MERCHANT_ID         VARCHAR(36)     NOT NULL,
    CUSTOMER_ID         VARCHAR(36)     NOT NULL,
    TRANSACTION_DATE    TIMESTAMP_NTZ   NOT NULL,
    AMOUNT              DECIMAL(12,2)   NOT NULL,
    CURRENCY            VARCHAR(3)      NOT NULL      COMMENT 'ISO 4217 currency code',
    PAYMENT_METHOD      VARCHAR(20)     NOT NULL      COMMENT 'CREDIT, DEBIT, ACH, WIRE',
    CARD_BRAND          VARCHAR(20)                   COMMENT 'VISA, MASTERCARD, AMEX, DISCOVER',
    CARD_LAST_FOUR      VARCHAR(4)                    COMMENT 'Last four digits only (PCI compliant)',
    STATUS              VARCHAR(20)     NOT NULL      COMMENT 'APPROVED, DECLINED, PENDING, REFUNDED',
    DECLINE_REASON      VARCHAR(100),
    AUTH_CODE           VARCHAR(10),
    RISK_SCORE          DECIMAL(5,2)                  COMMENT 'Real-time risk score 0-100',
    CHANNEL             VARCHAR(20)                   COMMENT 'ONLINE, IN_STORE, MOBILE, PHONE',
    REGION              VARCHAR(50),
    COUNTRY_CODE        VARCHAR(2),
    FEE_AMOUNT          DECIMAL(8,2)                  COMMENT 'Processing fee',
    CREATED_AT          TIMESTAMP_NTZ   NOT NULL      DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_TRANSACTIONS PRIMARY KEY (TRANSACTION_ID),
    CONSTRAINT FK_TXN_MERCHANT FOREIGN KEY (MERCHANT_ID) REFERENCES MERCHANTS(MERCHANT_ID),
    CONSTRAINT FK_TXN_CUSTOMER FOREIGN KEY (CUSTOMER_ID) REFERENCES CUSTOMERS(CUSTOMER_ID)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS IDX_TXN_DATE ON TRANSACTIONS (TRANSACTION_DATE);
CREATE INDEX IF NOT EXISTS IDX_TXN_MERCHANT ON TRANSACTIONS (MERCHANT_ID);
CREATE INDEX IF NOT EXISTS IDX_TXN_CUSTOMER ON TRANSACTIONS (CUSTOMER_ID);
CREATE INDEX IF NOT EXISTS IDX_TXN_STATUS ON TRANSACTIONS (STATUS);
CREATE INDEX IF NOT EXISTS IDX_TXN_METHOD ON TRANSACTIONS (PAYMENT_METHOD);

-- ============================================================
-- SETTLEMENTS - Merchant settlement batches
-- ============================================================

CREATE TABLE IF NOT EXISTS SETTLEMENTS (
    SETTLEMENT_ID       VARCHAR(36)     NOT NULL,
    MERCHANT_ID         VARCHAR(36)     NOT NULL,
    SETTLEMENT_DATE     DATE            NOT NULL,
    TRANSACTION_COUNT   INTEGER         NOT NULL,
    GROSS_AMOUNT        DECIMAL(14,2)   NOT NULL,
    FEE_AMOUNT          DECIMAL(10,2)   NOT NULL,
    NET_AMOUNT          DECIMAL(14,2)   NOT NULL,
    CURRENCY            VARCHAR(3)      NOT NULL,
    PAYMENT_METHOD      VARCHAR(20),
    STATUS              VARCHAR(20)     NOT NULL      COMMENT 'PENDING, COMPLETED, FAILED',
    SETTLEMENT_DAYS     INTEGER                       COMMENT 'Days from transaction to settlement',
    CREATED_AT          TIMESTAMP_NTZ   NOT NULL      DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_SETTLEMENTS PRIMARY KEY (SETTLEMENT_ID),
    CONSTRAINT FK_SET_MERCHANT FOREIGN KEY (MERCHANT_ID) REFERENCES MERCHANTS(MERCHANT_ID)
);

CREATE INDEX IF NOT EXISTS IDX_SET_DATE ON SETTLEMENTS (SETTLEMENT_DATE);
CREATE INDEX IF NOT EXISTS IDX_SET_MERCHANT ON SETTLEMENTS (MERCHANT_ID);

-- ============================================================
-- CHARGEBACKS - Dispute records
-- ============================================================

CREATE TABLE IF NOT EXISTS CHARGEBACKS (
    CHARGEBACK_ID           VARCHAR(36)     NOT NULL,
    TRANSACTION_ID          VARCHAR(36)     NOT NULL,
    MERCHANT_ID             VARCHAR(36)     NOT NULL,
    CHARGEBACK_DATE         DATE            NOT NULL,
    AMOUNT                  DECIMAL(12,2)   NOT NULL,
    CURRENCY                VARCHAR(3)      NOT NULL,
    REASON_CODE             VARCHAR(10)     NOT NULL,
    REASON_DESCRIPTION      VARCHAR(200),
    STATUS                  VARCHAR(20)     NOT NULL    COMMENT 'OPEN, WON, LOST, EXPIRED',
    RESOLUTION_DATE         DATE,
    RESOLUTION_DAYS         INTEGER                     COMMENT 'Days from filing to resolution',
    REPRESENTMENT_SUBMITTED BOOLEAN,
    CREATED_AT              TIMESTAMP_NTZ   NOT NULL    DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_CHARGEBACKS PRIMARY KEY (CHARGEBACK_ID),
    CONSTRAINT FK_CB_TRANSACTION FOREIGN KEY (TRANSACTION_ID) REFERENCES TRANSACTIONS(TRANSACTION_ID),
    CONSTRAINT FK_CB_MERCHANT FOREIGN KEY (MERCHANT_ID) REFERENCES MERCHANTS(MERCHANT_ID)
);

CREATE INDEX IF NOT EXISTS IDX_CB_DATE ON CHARGEBACKS (CHARGEBACK_DATE);
CREATE INDEX IF NOT EXISTS IDX_CB_MERCHANT ON CHARGEBACKS (MERCHANT_ID);
CREATE INDEX IF NOT EXISTS IDX_CB_STATUS ON CHARGEBACKS (STATUS);

-- ============================================================
-- Cortex Search Service for PCI Compliance RAG
-- ============================================================

-- Vector table for PCI document embeddings
CREATE TABLE IF NOT EXISTS PCI_DOCUMENT_CHUNKS (
    CHUNK_ID            VARCHAR(200)    NOT NULL,
    DOCUMENT_NAME       VARCHAR(200)    NOT NULL,
    SECTION_HEADING     VARCHAR(500),
    CHUNK_TEXT           TEXT            NOT NULL,
    CHUNK_INDEX         INTEGER,
    EMBEDDING           VECTOR(FLOAT, 384),
    CREATED_AT          TIMESTAMP_NTZ   NOT NULL      DEFAULT CURRENT_TIMESTAMP(),

    CONSTRAINT PK_PCI_CHUNKS PRIMARY KEY (CHUNK_ID)
);

-- Cortex Search Service definition (requires Snowflake Cortex)
-- CREATE CORTEX SEARCH SERVICE PCI_COMPLIANCE_SEARCH
--     ON PCI_DOCUMENT_CHUNKS
--     WAREHOUSE = PAYMENT_WH
--     TARGET_LAG = '1 hour'
--     EMBEDDING_MODEL = 'e5-base-v2'
--     AS (
--         SELECT
--             CHUNK_TEXT AS search_text,
--             DOCUMENT_NAME,
--             SECTION_HEADING,
--             CHUNK_ID
--         FROM PCI_DOCUMENT_CHUNKS
--     );

-- ============================================================
-- Warehouse Configuration
-- ============================================================

CREATE WAREHOUSE IF NOT EXISTS PAYMENT_WH
    WITH WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- ============================================================
-- Role and Grants
-- ============================================================

CREATE ROLE IF NOT EXISTS PAYMENT_ANALYST;

GRANT USAGE ON DATABASE PAYMENT_DB TO ROLE PAYMENT_ANALYST;
GRANT USAGE ON SCHEMA PAYMENT_SCHEMA TO ROLE PAYMENT_ANALYST;
GRANT SELECT ON ALL TABLES IN SCHEMA PAYMENT_SCHEMA TO ROLE PAYMENT_ANALYST;
GRANT USAGE ON WAREHOUSE PAYMENT_WH TO ROLE PAYMENT_ANALYST;
