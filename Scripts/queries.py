
#-----------------------------------------------------------#

query_baseline = """
CALL{MATCH (a)
SET a.in_A = CASE WHEN EXISTS {MATCH (a)-[:CAUSES*0..]->(z) WHERE z.name IN $Z_names} THEN TRUE ELSE FALSE END}
MATCH (a)-[r1]-(b)-[r2]-(c)
WHERE
EXISTS { MATCH (a)-[r1]->(b)-[r2]->(c) WHERE b.name IN $Z_names }
OR EXISTS { MATCH (a)<-[r1]-(b)<-[r2]-(c) WHERE b.name IN $Z_names }
OR EXISTS { MATCH (a)<-[r1]-(b)-[r2]->(c) WHERE b.name IN $Z_names }
OR EXISTS { MATCH (a)-[r1]->(b {in_A:FALSE})<-[r2]-(c)}
WITH COLLECT([r1, r2]) AS B
OPTIONAL MATCH (y)
WHERE NOT y.name IN $X_names + $Z_names
AND NOT EXISTS {
MATCH p = (x)-[:CAUSES]-+(y)
WHERE x.name IN $X_names
AND NONE(n IN nodes(p) WHERE size([m IN nodes(p) WHERE m = n]) > 1)
AND ALL(i IN RANGE(0, size(relationships(p))-2)
WHERE NOT [relationships(p)[i], relationships(p)[i+1]] IN B)
}
RETURN COLLECT(y.name) AS Y_all
"""

query_baseline_1 = """
CALL{MATCH (a)
SET a.in_A = CASE WHEN EXISTS {MATCH (a)-[:CAUSES*0..]->(z) WHERE z.name IN $Z_names} THEN TRUE ELSE FALSE END}
MATCH (a)-[r1]-(b)-[r2]-(c)
WHERE
EXISTS { MATCH (a)-[r1]->(b)-[r2]->(c) WHERE b.name IN $Z_names }
OR EXISTS { MATCH (a)<-[r1]-(b)<-[r2]-(c) WHERE b.name IN $Z_names }
OR EXISTS { MATCH (a)<-[r1]-(b)-[r2]->(c) WHERE b.name IN $Z_names }
OR EXISTS { MATCH (a)-[r1]->(b {in_A:FALSE})<-[r2]-(c)}
WITH COLLECT([r1, r2]) AS B
OPTIONAL MATCH (y)
WHERE NOT y.name IN $X_names + $Z_names
AND NOT EXISTS {
MATCH p = (x)-[:CAUSES]-{1,"""

query_baseline_2 = """}(y)
WHERE x.name IN $X_names
AND NONE(n IN nodes(p) WHERE size([m IN nodes(p) WHERE m = n]) > 1)
AND ALL(i IN RANGE(0, size(relationships(p))-2)
WHERE NOT [relationships(p)[i], relationships(p)[i+1]] IN B)
}
RETURN COLLECT(y.name) AS Y_all
"""

#-----------------------------------------------------------#

query_dcollision_reset = """
// Step 0: Reset
CALL(){MATCH ()-[r]->()
WHERE r:CONNECTED
DELETE r}
CALL(){MATCH (n)
REMOVE n.in_X, n.in_Z, n.in_A, n.pc, n.candidate}
"""

query_dcollision_partial_reset = """
MATCH (n)
REMOVE n.in_X, n.pc, n.candidate
"""

#-----------------------------------------------------------#

query_dcollision_1of4 = """
MATCH (n) 
SET n.in_Z = CASE WHEN n.name IN $Z_names THEN TRUE ELSE FALSE END,
n.in_A = FALSE
"""

query_dcollision_2of4 = """
// Step 1: Find the ancestors of the nodes in Z, and set the properties
MATCH (a)
WHERE EXISTS {(a)-[:CAUSES*0..]->({in_Z:TRUE})}
SET a.in_A = TRUE
"""

query_dcollision_3of4 =  """
// Step 2: Draw the :CONNECTED edges determined by Z
CALL() {
MATCH (a1 {in_Z:FALSE})-[:CAUSES]->(c1 {in_A:FALSE})<-[:CAUSES]-(b1 {in_Z:FALSE})
WHERE elementId(a1) < elementId(b1)
MERGE (a1)-[:CONNECTED]->(c1)
MERGE (b1)-[:CONNECTED]->(c1)
}
CALL() {
MATCH (a2 {in_Z:FALSE})-[:CAUSES]->(b2)
WHERE NOT EXISTS {MATCH (a2)-[:CONNECTED]->(b2)}
MERGE (a2)-[:CONNECTED]->(b2)
MERGE (b2)-[:CONNECTED]->(a2)
}
"""

#-----------------------------------------------------------#

query_dcollision_4of4 = """
// Step 3: Find the nodes which are d-connected to X

MATCH (n)
SET n.in_X = CASE WHEN n.name IN $X_names THEN TRUE ELSE FALSE END,
n.candidate = CASE WHEN n.name IN $Z_names + $X_names THEN FALSE ELSE TRUE END,
n.pc = CASE WHEN n.name IN $X_names THEN TRUE ELSE FALSE END
WITH n WHERE n.candidate=TRUE
WITH COLLECT(n) AS Candidates

CALL {WITH Candidates
MATCH (n) WHERE n IN Candidates
AND EXISTS {(n)-[:CONNECTED*1..]->(x {in_X:TRUE})}
SET n.candidate = FALSE, n.pc = TRUE}

CALL() {MATCH (n {candidate:TRUE})
WHERE EXISTS {(n)<-[:CONNECTED*1..]-(c {pc:TRUE})}
SET n.candidate = FALSE}

// Step 4: obtain the complementary set of d_connected
MATCH (n {candidate:TRUE})
RETURN COLLECT(n.name) AS all_d_separated
"""

query_dcollision_4of4_1 = """
// Step 3: Find the nodes which are d-connected to X

MATCH (n)
SET n.in_X = CASE WHEN n.name IN $X_names THEN TRUE ELSE FALSE END,
n.candidate = CASE WHEN n.name IN $Z_names + $X_names THEN FALSE ELSE TRUE END,
n.pc = CASE WHEN n.name IN $X_names THEN TRUE ELSE FALSE END
WITH n WHERE n.candidate=TRUE
WITH COLLECT(n) AS Candidates

CALL {WITH Candidates
MATCH (n) WHERE n IN Candidates
AND EXISTS {(n)-[:CONNECTED]->{1,"""
                   
query_dcollision_4of4_2 = """}(x {in_X:TRUE})}
SET n.candidate = FALSE, n.pc = TRUE}

CALL() {MATCH (n {candidate:TRUE})
WHERE EXISTS {(n)<-[:CONNECTED]-{1,"""
    
query_dcollision_4of4_3 = """}(c {pc:TRUE})}
SET n.candidate = FALSE}

// Step 4: obtain the complementary set of d_connected
MATCH (n {candidate:TRUE})
RETURN COLLECT(n.name) AS all_d_separated
"""


#-----------------------------------------------------------#

query_dcollision_4of4_apoc = """
MATCH (n)
SET n.in_X = CASE WHEN n.name IN $X_names THEN TRUE ELSE FALSE END,
n.candidate = CASE WHEN n.name IN $Z_names + $X_names THEN FALSE ELSE TRUE END,
n.pc = CASE WHEN n.name IN $X_names THEN TRUE ELSE FALSE END
WITH n WHERE n.candidate=TRUE
WITH COLLECT(n) AS Candidates


CALL {WITH Candidates
MATCH (x {in_X:TRUE})
CALL apoc.path.subgraphAll(x, {relationshipFilter:'<CONNECTED', minLevel:1, endNodes:Candidates})
YIELD nodes
UNWIND nodes AS n
WITH DISTINCT n
SET n.candidate=FALSE, n.pc=TRUE}


CALL() {MATCH (n {candidate: TRUE})
WITH COLLECT(n) AS Candidates
MATCH (c {pc:TRUE})
CALL apoc.path.subgraphAll(c, {relationshipFilter:'CONNECTED>', minLevel:1, endNodes:Candidates})
YIELD nodes
UNWIND nodes AS n
WITH DISTINCT n
SET n.candidate = FALSE}


// Step 4: obtain the complementary set of d_connected
MATCH (n {candidate:TRUE})
RETURN COLLECT(n.name) AS all_d_separated
"""

#-----------------------------------------------------------#