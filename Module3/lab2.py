from sentence_transformers import SentenceTransformer, util

# Chargement du modèle
model = SentenceTransformer("all-MiniLM-L6-v2")

# Phrases
s1 = "assurance habitation"
s2 = "contrat d’assurance maison"
s3 = "voiture électrique"

# Génération des embeddings
e1, e2, e3 = model.encode([s1, s2, s3])

# Calcul des similarités cosinus
sim_12 = util.cos_sim(e1, e2)
sim_13 = util.cos_sim(e1, e3)
sim_23 = util.cos_sim(e2, e3)

print("s1 vs s2 :", sim_12)
print("s1 vs s3 :", sim_13)
print("s2 vs s3 :", sim_23)
