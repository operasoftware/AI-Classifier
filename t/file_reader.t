use strict;
use warnings;

use Test::More;
use AI::Classifier::Text::FileLearner;

my $iterator = AI::Classifier::Text::FileLearner->new( training_dir => 't/data/training_set_ordered/' );


my %hash;
while( my $doc = $iterator->next ){
    $hash{$doc->{file}} = $doc;
}
my $target = {
    't/data/training_set_ordered/spam/1' => {
        'features' => { ccccc => 1, NO_URLS => 2 },
        'file' => 't/data/training_set_ordered/spam/1',
        'categories' => [ 'spam' ]
    },
    't/data/training_set_ordered/ham/2' => {
        'features' => { ccccc => 1, aaaa => 1, NO_URLS => 2 },
        'file' => 't/data/training_set_ordered/ham/2',
        'categories' => [ 'ham' ]
    }
};
is_deeply( \%hash, $target );

my $classifier = AI::Classifier::Text::FileLearner->new( training_dir => 't/data/training_set_ordered/' )->classifier;

ok( $classifier, 'Classifier created' );
ok( $classifier->classifier->model()->{prior_probs}{ham}, 'ham prior probs' );
ok( $classifier->classifier->model()->{prior_probs}{spam}, 'spam prior probs' );
{
    my $iterator = AI::Classifier::Text::FileLearner->new( training_dir => 't/data/training_initial_features/' );

    my %hash;
    while( my $doc = $iterator->next ){
        $hash{$doc->{file}} = $doc;
    }
    my $target = {
        't/data/training_initial_features/ham/1' => {
            'file' => 't/data/training_initial_features/ham/1',
            'categories' => [ 'ham' ],
            features => { trala => 1, some_tag => 3, NO_URLS => 2 }
        },
    };
    is_deeply( \%hash, $target );
}


done_testing;

