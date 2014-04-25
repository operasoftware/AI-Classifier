package AI::Classifier::Text;

use strict;
use warnings;
use 5.010;
use Moose;
use MooseX::Storage;

use AI::Classifier::Text::Analyzer;
use Module::Load (); # don't overwrite our sub load() with Module::Load::load()

with Storage(format => 'Storable', io => 'File');

has classifier => (is => 'ro', required => 1 );
has analyzer => ( is => 'ro', default => sub{ AI::Classifier::Text::Analyzer->new() } );
# for store/load only, don't touch unless you really know what you're doing
has classifier_class => (is => 'bare');

around BUILDARGS => sub {
    my $orig  = shift;
    my $class = shift;
    my %args = @_;

    if( $args{ training_data } ){
        require AI::NaiveBayes::Learner;
        my $learner = AI::NaiveBayes::Learner->new(features_kept => 0.5);
        for my $example ( @{ $args{ training_data } } ){
            $learner->add_example( %$example );
        }
        my $classifier = $learner->classifier;
        $args{ classifier } = $classifier;
        delete $args{ training_data };
    }
    return $class->$orig( %args );
};

before store => sub {
    my $self = shift;
    $self->{classifier_class} = $self->classifier->meta->name;
};

around load => sub {
    my ($orig, $class) = (shift, shift);
    my $self = $class->$orig(@_);
    Module::Load::load($self->{classifier_class});
    return $self;
};

sub classify {
    my( $self, $text, $features ) = @_;
    return $self->classifier->classify( $self->analyzer->analyze( $text, $features ) );
}

__PACKAGE__->meta->make_immutable;

1;

__END__

# ABSTRACT: A convenient class for text classification

=head1 SYNOPSIS

    my $cl = AI::Classifier::Text->new(
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
    );
    # the above creates a default AI::NaiveBayes classifier and feeds it the training data

    my $res = $cl->classify("I would like to begin farming sheep" );

    $res    = $cl->classify("I would like to begin farming sheep", { new_user => 1 });

    print $res->best_category; 
    $cl->store('some-file');
    # later
    my $cl = AI::Classifier::Text->load('some-file');
    my $res = $cl->classify("do cats eat sheep?");

=head1 DESCRIPTION

C<AI::Classifier::Text> combines a lexical analyzer (by default being
L<AI::Classifier::Text::Analyzer>) and a compatible classifier to perform text classification.

The constructor requires either a compatible trained classifier (like C<AI::NaiveBayes>) - or
training_data parameter with a list of training examples.
In that later case it creates the default
C<AI::NaiveBayes> classifier and traubs it before constructing the C<AI::Classifier::Text> object.

If your training data does not feet into the computer memory, 
or you want a different classifier to use - than train the classifier first and then pass 
it to the C<AI::Classifier::Text> constructor.


This is partially based on C<AI::TextCategorizer>.

=head1 ATTRIBUTES

=over 4

=item C<classifier>

An object that'll perform classification of supplied feature vectors. Has to
define a C<classify()> method, which accepts a hash refence. The return value of
AI::Classifier::Text->classify() will be the return value of C<classifier>'s
C<classify()> method.

This attribute has to be supplied to the C<new()> method during object creation.

=item C<analyzer>

The class performing lexical analysis of the text in order to produce a feature
vector. This defaults to C<AI::Classifier::Text::Analyzer>.

=back

=head1 METHODS

=over 4

=item C<< new(classifier => $foo) >>

Creates a new C<AI::Classifier::Text> object. It requires either the classifier 
or the training data passed to it.

=item C<classify($document, $features)>

Categorize the given document. A lexical analyzer will be used to extract
features from C<$document>, and in addition to that the features from
C<$features> hash reference will be added. The return value comes directly from
the C<classifier> object's C<classify> method.

=back

=head1 SEE ALSO

AI::NaiveBayes (3), AI::Categorizer(3)

=cut
