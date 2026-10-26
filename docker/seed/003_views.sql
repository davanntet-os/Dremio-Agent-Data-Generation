-- Documented joins as views (verified relationship evidence for dataagent).
CREATE VIEW v_customer_orders AS
SELECT c.customer_id, c.first_name, c.last_name, o.order_id, o.order_date, o.status, o.total_amount
FROM public.customers c
JOIN public.orders o ON o.customer_id = c.customer_id;

CREATE VIEW v_order_item_returns AS
SELECT oi.order_id, oi.product_id, oi.quantity, oi.line_amount, r.return_id, r.return_date, r.refund_amount
FROM public.order_items oi
JOIN public.returns r ON r.order_item_id = oi.order_item_id;

CREATE VIEW v_revenue_by_region AS
SELECT c.region, SUM(o.total_amount) AS region_revenue
FROM public.customers c
JOIN public.orders o ON o.customer_id = c.customer_id
WHERE o.status = 'COMPLETED'
GROUP BY c.region;
