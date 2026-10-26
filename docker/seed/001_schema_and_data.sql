-- Synthetic e-commerce seed schema for dataagent development & testing.
-- Runs once on first Postgres init (docker-entrypoint-initdb.d).

CREATE TABLE customers (
    customer_id   BIGINT PRIMARY KEY,
    first_name    VARCHAR(64),
    last_name     VARCHAR(64),
    email         VARCHAR(128) UNIQUE,
    signup_date   TIMESTAMP,
    region        VARCHAR(32),
    loyalty_tier  VARCHAR(16)
);

CREATE TABLE products (
    product_id    BIGINT PRIMARY KEY,
    product_name  VARCHAR(128),
    category      VARCHAR(64),
    unit_price    NUMERIC(10, 2)
);

CREATE TABLE orders (
    order_id       BIGINT PRIMARY KEY,
    customer_id    BIGINT REFERENCES customers(customer_id),
    order_date     TIMESTAMP,
    status         VARCHAR(16),
    total_amount   NUMERIC(12, 2),
    shipping_cost  NUMERIC(8, 2)   -- intentionally sparse: NULL for digital orders
);

CREATE TABLE order_items (
    order_item_id  BIGINT PRIMARY KEY,
    order_id       BIGINT REFERENCES orders(order_id),
    product_id     BIGINT REFERENCES products(product_id),
    quantity       INTEGER,
    line_amount    NUMERIC(12, 2)
);

CREATE TABLE returns (
    return_id      BIGINT PRIMARY KEY,
    order_item_id  BIGINT REFERENCES order_items(order_item_id),
    return_date    TIMESTAMP,
    reason         VARCHAR(64),   -- intentionally sparse: NULL when not stated
    refund_amount  NUMERIC(12, 2)
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date     ON orders(order_date);
CREATE INDEX idx_items_order     ON order_items(order_id);
CREATE INDEX idx_returns_item    ON returns(order_item_id);

COMMENT ON TABLE customers IS 'Customer master records with region and loyalty tier';
COMMENT ON COLUMN customers.customer_id IS 'Unique customer identifier';
COMMENT ON COLUMN customers.signup_date IS 'Timestamp when the customer account was created';
COMMENT ON TABLE products IS 'Product catalog with category and list price';
COMMENT ON COLUMN products.unit_price IS 'Current list price per single unit';
COMMENT ON TABLE orders IS 'Order headers, one row per order with status and totals';
COMMENT ON COLUMN orders.status IS 'Order lifecycle status: PENDING, COMPLETED, CANCELLED or RETURNED';
COMMENT ON COLUMN orders.total_amount IS 'Order total including taxes, excluding shipping';
COMMENT ON COLUMN orders.shipping_cost IS 'Shipment cost; NULL for digital-only orders';
COMMENT ON TABLE order_items IS 'Order line items linking orders to products';
COMMENT ON COLUMN order_items.line_amount IS 'quantity * unit_price at time of purchase';
COMMENT ON TABLE returns IS 'Returned order items with refund information';
COMMENT ON COLUMN returns.reason IS 'Customer-stated return reason; NULL when not provided';

-- ------------------------------------------------------------------
-- Seed data (deterministic pseudo-random via generate_series)
-- ------------------------------------------------------------------

INSERT INTO customers
SELECT   g                                          AS customer_id,
         'First' || g                               AS first_name,
         'Last'  || g                               AS last_name,
         'customer' || g || '@example.com'          AS email,
         TIMESTAMP '2022-01-10 09:00:00'
             + (g * interval '7 hours')
             + ((g % 60) * interval '1 day')        AS signup_date,
         (ARRAY['NORTH', 'SOUTH', 'EAST', 'WEST'])[1 + g % 4] AS region,
         (ARRAY['NONE', 'BRONZE', 'SILVER', 'GOLD'])[1 + g % 4] AS loyalty_tier
FROM generate_series(1, 500) g;

INSERT INTO products
SELECT   g                                          AS product_id,
         'Product ' || g                            AS product_name,
         (ARRAY['ELECTRONICS', 'HOME', 'GARDEN',
                'TOYS', 'SPORTS', 'BOOKS'])[1 + g % 6] AS category,
         round((10 + (g * 37) % 990)::numeric, 2)   AS unit_price
FROM generate_series(1, 120) g;

INSERT INTO orders
SELECT   g                                          AS order_id,
         1 + ((g * 13) % 500)                       AS customer_id,
         TIMESTAMP '2023-06-01 08:00:00'
             + ((g % 500) * interval '1 day')
             + ((g % 24) * interval '1 hour')       AS order_date,
         (ARRAY['PENDING', 'COMPLETED', 'COMPLETED', 'COMPLETED',
                'CANCELLED', 'COMPLETED'])[1 + g % 6] AS status,
         round((20 + (g * 53) % 4800)::numeric, 2)  AS total_amount,
         CASE WHEN g % 9 = 0 THEN NULL              -- digital orders: no shipping
              ELSE round((3 + (g % 15))::numeric, 2) END AS shipping_cost
FROM generate_series(1, 5000) g;

INSERT INTO order_items
SELECT   g                                          AS order_item_id,
         1 + ((g * 7) % 5000)                       AS order_id,
         1 + ((g * 3) % 120)                        AS product_id,
         1 + (g % 5)                                AS quantity,
         round((10 + (g * 41) % 2500)::numeric, 2)  AS line_amount
FROM generate_series(1, 11000) g;

INSERT INTO returns
SELECT   g                                          AS return_id,
         1 + ((g * 11) % 11000)                     AS order_item_id,
         TIMESTAMP '2023-07-01 10:00:00'
             + ((g % 400) * interval '1 day')       AS return_date,
         CASE WHEN g % 5 = 0 THEN NULL              -- reason sometimes not stated
              ELSE (ARRAY['DAMAGED', 'WRONG ITEM', 'NOT AS DESCRIBED', 'CHANGED MIND'])[1 + g % 4] END AS reason,
         round((10 + (g * 29) % 800)::numeric, 2)   AS refund_amount
FROM generate_series(1, 700) g;
