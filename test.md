curl -X POST http://localhost:3000/calculate \
     -H "Content-Type: application/json" \
     -d '{
           "attacker": {
             "name": "Gengar",
             "item": "Choice Specs",
             "nature": "Timid",
             "evs": {"spa": 252},
             "level": 50
           },
           "defender": {
             "name": "Chansey",
             "item": "Eviolite",
             "nature": "Calm",
             "evs": {"hp": 252, "spd": 252},
             "level": 50
           },
           "move": "Focus Blast"
         }'