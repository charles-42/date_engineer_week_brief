
-- On cherche à calculer le nombre de commandes, le nombre de produits vendus 
-- et le chiffre d'affaires (qui n’intègre pas les frais de port) 
-- pour chaque état du client. 

-- Pour cela, à l’aide d’une requête imbriquée ou d’une table temporaire procédez en trois temps:
-- à partir de la table orders items calculez les indicateurs souhaités
-- faire une jointure ensuite avec la table order pour récupérer la date de vente
-- faire une jointure enfin sur la table consumers pour récupérer le pays.


WITH order_price AS (
SELECT order_id, COUNT(*) AS nb_item, sum(price) AS total_price
FROM OrderItem
GROUP BY order_id
)
SELECT C.customer_state, COUNT(A.order_id) as nb_commandes, SUM(A.nb_item) as nb_produits, SUM(A.total_price) as chiffre_affaires
FROM order_price A
LEFT JOIN Orders B USING (order_id)
LEFT JOIN Customers C ON B.customer_id = C.customer_id
GROUP BY C.customer_state
;