File = cell(10, 1);
File{1} = 'ionosphere_train';
File{2} = 'ionosphere_test';
File{3} = 'isolet_train';
File{4} = 'isolet_test';
File{5} = 'liver_train';
File{6} = 'liver_test';
File{7} = 'mnist_train';
File{8} = 'mnist_test';
File{9} = 'mushroom_train';
File{10} = 'mushroom_test';

for fid = 1:10
  load(File{fid});
  X = full(X);
  fout = fopen(strcat(File{fid}, '.dat'), 'w');
  for i = 1:size(X, 1)
    fprintf(fout, '%i', Y(i) * 2 - 1);
    for j = 1:size(X, 2)
      fprintf(fout, ' %i:%f', j, X(i, j));
    end
    fprintf(fout, '\n');
  end
  fclose(fout);
end
