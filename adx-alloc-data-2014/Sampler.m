
samples = 1e5;
tau = 0;


for i = 1:7
    adfile = sprintf('pub%d-ads.txt', i)
    typefile = sprintf('pub%d-types.txt', i)
    samplefile = sprintf('pub%d-sample.txt', i)
    [ A, T, rho, type_prob, type_ad, type ] = LoadSynthFile( adfile, typefile );

    fprintf('Loading...\n');
    [Q, t] = GenerateSample (A, T, rho, type_ad, type_prob, tau, type, samples);

    fprintf('Saving...\n');
    csvwrite(samplefile,Q);
end