
use strict;
use Test::More tests => 6;
use AI::Classifier::Text;
ok(1); # If we made it this far, we're loaded.

my $text_classifier = AI::Classifier::Text->new( 
    training_data => [
        {
            attributes => _hash(qw(sheep very valuable farming)),
            labels => ['farming']
        },
        {
            attributes => _hash(qw(farming requires many kinds animals)),
            labels => ['farming']
        },
        {
            attributes => _hash(qw(vampires drink blood vampires may staked)),
            labels => ['vampire']
        },
        {
            attributes => _hash(qw(vampires cannot see their images mirrors)),
            labels => ['vampire']
        },
    ],
);

isa_ok( $text_classifier, 'AI::Classifier::Text' );
isa_ok( $text_classifier->classifier, 'AI::NaiveBayes' );
isa_ok( $text_classifier->analyzer, 'AI::Classifier::Text::Analyzer' );

# Predict
my $s = $text_classifier->classify( "I would like to begin farming sheep" );
isa_ok( $s, 'AI::NaiveBayes::Classification' );
is( $s->best_category, 'farming' );


################################################################
sub _hash { +{ map {$_,1} @_ } }

