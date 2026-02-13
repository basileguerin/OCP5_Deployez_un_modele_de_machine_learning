-- Extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

DROP TABLE IF EXISTS prediction_results CASCADE;
DROP TABLE IF EXISTS prediction_requests CASCADE;
DROP TABLE IF EXISTS employees CASCADE;

-- CREATE

-- Dataset principal
CREATE TABLE employees (
  employee_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  features JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Inputs envoyés au modèle
CREATE TABLE prediction_requests (
  request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  input_features JSONB NOT NULL,
  requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Outputs du modèle
CREATE TABLE prediction_results (
  result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  request_id UUID NOT NULL UNIQUE REFERENCES prediction_requests(request_id) ON DELETE CASCADE,
  probability DOUBLE PRECISION NOT NULL,
  prediction INTEGER NOT NULL,
  threshold DOUBLE PRECISION NOT NULL,
  predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index
CREATE INDEX idx_employees_features_gin ON employees USING GIN (features);
CREATE INDEX idx_predres_predicted_at ON prediction_results(predicted_at);
