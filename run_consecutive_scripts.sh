#!/bin/bash
python main_run.py --dof vel --src_dom C &
python main_run.py --dof vel --src_dom B &
python main_run.py --dof pos --src_dom B --no_epochs 401  --total_budget 281 &
python main_run.py --dof pos --src_dom C --no_epochs 401 --total_budget 281 &
wait