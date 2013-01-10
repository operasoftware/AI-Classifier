package AI::Classifier::Text::FileLearner;
use strict;
use warnings;
use 5.010;

use Moose;
use File::Find::Rule;
use List::Util 'max';
use Carp 'croak';
use AI::NaiveBayes::Learner;
use AI::Classifier::Text;
use AI::Classifier::Text::Analyzer;

has term_weighting => (is => 'ro', isa => 'Str');
has analyzer => ( is => 'ro', default => sub{ AI::Classifier::Text::Analyzer->new() } );
has learner => ( is => 'ro', default => sub{ AI::NaiveBayes::Learner->new() } );
has training_dir => ( is => 'ro', isa => 'Str', required => 1 );
has iterator => ( is => 'ro', lazy_build => 1 );
sub _build_iterator {
    my $self = shift;
    my $rule = File::Find::Rule->new( );
    $rule->file;
    $rule->not_name('*.data');
    $rule->start( $self->training_dir );
    return $rule;
}

sub next {
    my $self = shift;

    my $file = $self->iterator->match;
    return if !defined($file);
    my $training_dir = $self->training_dir;
    $file =~ /^$training_dir(.*?)\//;
    my $category = $1;
    open(my $fh, "<:encoding(UTF-8)", $file )
    || Carp::croak(
                "Unable to read the specified training file: $file\n");
    my $content = join('', <$fh>);
    close $fh;
    my $initial_features = {};
    if( -f "$file.data" ){
        my $data = do "$file.data";
        $initial_features = $data->{initial_features}
    }
    my $features = $self->analyzer->analyze( $content, $initial_features );

    return { 
        file => $file, 
        features => $features, 
        categories => [ $category ],
    };
}

sub internal_classifier {
    my $self = shift;
    my $learner = $self->learner;
    while ( my $data  = $self->next ) {
        normalize( $data->{features} );
        if (defined $self->term_weighting) {
            weight_terms($data, type => $self->term_weighting);
        }
        $learner->add_example( 
            attributes => $data->{features},
            labels     => $data->{categories}
        );
    }

    return $learner->classifier();
}


sub classifier {
    my $self = shift;
    return AI::Classifier::Text->new(
        classifier => $self->internal_classifier,
        analyzer => $self->analyzer,
    );
}


sub weight_terms {
    my ($doc, %args) = @_;
    my $f = $doc->{features};
    given ($args{type}) {
        when ('n') {
            my $max_tf = max values %$f;
            $_ = 0.5 + 0.5 * $_ / $max_tf for values %$f;
        }
        when ('b') {
            $_ = $_ ? 1 : 0 for values %$f;
        }
        default {
            croak 'Unknown weighting type: '.$args{type};
        }
    }
}

# this doesn't quite fit the current model (it requires the entire collection
# of documents to be in memory at once), but it may be useful to someone, someday
# so let's just leave it here
sub collection_weighting {
    my (@documents, $subtrahend) = @_;
    $subtrahend //= 0;

    my $num_docs   = +@documents;

    my %frequency;
    for my $doc (@documents) {
        for my $k (keys %{$doc->{attributes}}) {
            $frequency{$k}++;
        }
    }

    foreach my $doc (@documents) {
        my $f = $doc->{attributes};
        for (keys %$f) {
            $f->{$_} *= log($num_docs / ($frequency{$_} // 0) - $subtrahend);
        }
    }
}

sub euclidean_length {
    my $f = shift;

    my $total = 0;
    foreach (values %$f) {
        $total += $_**2;
    }

    return sqrt($total);
}

sub scale {
    my ($f, $scalar) = @_;

    $_ *= $scalar foreach values %$f;

    return $f;
}

sub normalize {
    my $attrs = shift;

    my $length = euclidean_length($attrs);

    return $length ? scale($attrs, 1/$length) : $attrs;
}

1;

__END__

# ABSTRACT: Training data reader for AI::NaiveBayes

=head1 SYNOPSIS

    use AI::Classifier::Text::FileLearner;

    my $iterator = AI::Classifier::Text::FileLearner->new( training_dir => 't/data/training_set_ordered/' );

    my $classifier = $iterator->classifier;

=head1 DESCRIPTION

This is a trainer of text classifiers.  It traverses a directory filled,
interprets the subdirectories in it as category names, reads all files in them and adds them
as examples for the classifier being trained.

head1 METHODS

=over 4

=item next

Internal method for traversing the training data directory.

=item classifier

Returns a trained classifier.

=back

=cut
