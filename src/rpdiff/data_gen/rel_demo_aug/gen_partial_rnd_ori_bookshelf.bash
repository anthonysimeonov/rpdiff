set -eo pipefail

if [ -z $1 ]
then
    SEED=0
    PORT=6000
else
    SEED=$1
    PORT=$2
fi

set -euo pipefail

printf "\nRunning with seed $SEED\n"
sleep 3

python bookshelf_demos.py --parent_class syn_bookshelf --child_class syn_book --exp book_on_bookshelf_double_view_rnd_ori --task_name book_in_bookshelf --pybullet_server -c bookshelf_double_view.yaml --num_iterations 1000 --seed $SEED -p $PORT --child_load_pose_type any_pose
# python bookshelf_demos.py --parent_class syn_bookshelf --child_class syn_book --exp book_on_bookshelf_double_view_half_rnd_ori_half_flat --task_name book_in_bookshelf --pybullet_server -c bookshelf_double_view.yaml --num_iterations 1000 --seed $SEED -p $PORT --load_pose_type any_pose
