CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Dataset principal
CREATE TABLE IF NOT EXISTS employees (
  employee_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  features JSONB NOT NULL,

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Inputs envoyés au modèle (via l’API)
CREATE TABLE IF NOT EXISTS prediction_requests (
  request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  employee_id UUID NULL REFERENCES employees(employee_id),
  input_features JSONB NOT NULL,
  requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Outputs du modèle
CREATE TABLE IF NOT EXISTS prediction_results (
  result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  request_id UUID NOT NULL UNIQUE REFERENCES prediction_requests(request_id) ON DELETE CASCADE,
  probability DOUBLE PRECISION NOT NULL,
  prediction INTEGER NOT NULL,
  threshold DOUBLE PRECISION NOT NULL,
  model_version TEXT NULL,
  predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index utiles
CREATE INDEX IF NOT EXISTS idx_employees_features_gin ON employees USING GIN (features);
CREATE INDEX IF NOT EXISTS idx_predreq_employee_id ON prediction_requests(employee_id);
CREATE INDEX IF NOT EXISTS idx_predres_predicted_at ON prediction_results(predicted_at);
