import csv

with open("company_policies.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    policies = list(reader)

print("Number of policies:", len(policies))
print()

print("First policy:")
print("Title:", policies[0]["title"])
print("Department:", policies[0]["department"])
print("Category:", policies[0]["category"])
print("Policy text:", policies[0]["policy_text"])
