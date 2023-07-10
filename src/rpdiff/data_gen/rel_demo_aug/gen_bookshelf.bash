if [ -z $1 ]
then
    SEED=0
else
    SEED=$1
fi

printf "\nRunning with seed $SEED\n"
sleep 3

python bookshelf_demos.py --parent_class syn_bookshelf --child_class syn_book --rel_demo_exp book_on_bookshelf --exp book_on_bookshelf_large --task_name book_in_bookshelf --pybullet_server --config bookshelf.yaml --num_iterations 1000 --seed $SEED
