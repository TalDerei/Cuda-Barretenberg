#pragma once

#include <array>
#include <vector>

#include "../../types/polynomial_manifest.hpp"
#include "../../types/prover_settings.hpp"
#include "../../types/program_witness.hpp"
#include "../../proving_key/proving_key.hpp"
#include "../../verification_key/verification_key.hpp"
#include "../../prover/work_queue.hpp"

namespace waffle {

namespace widget {
enum ChallengeIndex {
    ALPHA,
    BETA,
    GAMMA,
    ETA,
    ZETA,
    MAX_NUM_CHALLENGES,
};

/**
 * Widgets use this bitmask to declare the challenges
 * they will be using
 * */
#define CHALLENGE_BIT_ALPHA (1 << widget::ChallengeIndex::ALPHA)
#define CHALLENGE_BIT_BETA (1 << widget::ChallengeIndex::BETA)
#define CHALLENGE_BIT_GAMMA (1 << widget::ChallengeIndex::GAMMA)
#define CHALLENGE_BIT_ETA (1 << widget::ChallengeIndex::ETA)
#define CHALLENGE_BIT_ZETA (1 << widget::ChallengeIndex::ZETA)

namespace containers {
template <class Field, size_t num_widget_relations> struct challenge_array {
    std::array<Field, ChallengeIndex::MAX_NUM_CHALLENGES> elements;
    std::array<Field, num_widget_relations> alpha_powers;
};

template <class Field> using poly_array = std::array<std::pair<Field, Field>, PolynomialIndex::MAX_NUM_POLYNOMIALS>;

template <class Field> struct poly_ptr_array {
    std::array<Field*, PolynomialIndex::MAX_NUM_POLYNOMIALS> coefficients;
    size_t block_mask;
};

template <class Field> using coefficient_array = std::array<Field, PolynomialIndex::MAX_NUM_POLYNOMIALS>;

} // namespace containers

namespace getters {
template <class Field, class Transcript, class Settings, size_t num_widget_relations> class BaseGetter {
  protected:
    typedef containers::challenge_array<Field, num_widget_relations> challenge_array;

  public:
    /**
     * Create a challenge array from transcript.
     * Loads alpha, beta, gamma, eta, zeta and nu and calculates powers of alpha.
     *
     * @param transcript Transcript to get challenges from.
     * @param alpha_base alpha to some power (depends on previously used widgets).
     * @param required_challenges Challenge bitmask, which shows when the function should fail.
     *
     * @return A structure with an array of challenge values and powers of alpha.
     * */
    static challenge_array get_challenges(const Transcript& transcript,
                                          const Field& alpha_base,
                                          uint8_t required_challenges)

    {
        challenge_array result{};
        /**
         * There are several issues we need to address here:
         * 1. We can't just set the value to 0. In case of a typo this could lead to a vulnerability
         * 2. We can't fail when there is no challenge, because getters get activated at various phases
         * 3. There is no way for us to check accesses in the challenge_array (it would degrade speed)
         *
         * One of the mitigations we use is we force the transition widget kernel to have members
         * that describe the necessary challenges for quotient polynomial construction and for
         * kate update. We then take them and submit to the get_challenges function. This allows us
         * to catch misuse, but only if the developer is prudent.
         *
         * Since we can't enforce anything really we introduced a simple mitigation:
         *   The challenges that aren't in the transcript are replaced by random values.
         *
         * The value each of the widget uses and the value the verifier uses will differ. As a result,
         * proof will fail if some widget uses an uninitialized challenge.         *
         *
         * */
        auto add_challenge = [transcript,
                              &result](const auto label, const auto tag, const bool required, const size_t index = 0) {
            if (required) { // if we are supposed to use the challenge, we pick it from transcript, otherwise we use a
                            // random value
                ASSERT(index < transcript.get_num_challenges(
                                   label)); // We fail this assertion only if the required challenge doesn't exist yet
                result.elements[tag] = transcript.get_challenge_field_element(label, index);
            } else {
                result.elements[tag] = barretenberg::fr::random_element();
            }
        };

        add_challenge("alpha", ALPHA, required_challenges & CHALLENGE_BIT_ALPHA);
        add_challenge("beta", BETA, required_challenges & CHALLENGE_BIT_BETA);
        add_challenge("beta", GAMMA, required_challenges & CHALLENGE_BIT_GAMMA, 1);
        add_challenge("eta", ETA, required_challenges & CHALLENGE_BIT_ETA);
        add_challenge("z", ZETA, required_challenges & CHALLENGE_BIT_ZETA);
        result.alpha_powers[0] = alpha_base;
        for (size_t i = 1; i < num_widget_relations; ++i) {
            result.alpha_powers[i] = result.alpha_powers[i - 1] * result.elements[ALPHA];
        }
        return result;
    }

    static Field update_alpha(const challenge_array& challenges, const size_t num_independent_relations)
    {
        if (num_independent_relations == 0) {
            return challenges.alpha_powers[0];
        }
        return challenges.alpha_powers[num_independent_relations - 1] * challenges.elements[ALPHA];
    }
};

template <class Field, class Transcript, class Settings, size_t num_widget_relations>
class EvaluationGetter : public BaseGetter<Field, Transcript, Settings, num_widget_relations> {
  protected:
    typedef containers::poly_array<Field> poly_array;
    typedef std::vector<PolynomialDescriptor> polynomial_manifest;

  public:
    /**
     * Get a polynomial at offset id
     *
     * @param polynomials An array of polynomials
     * @param size_t Unused
     *
     * @tparam use_shifted_evaluation Whether to pick first or second
     * @tparam id Polynomial index.
     *
     * @return The chosen polynomial
     * */
    template <bool use_shifted_evaluation, PolynomialIndex id>
    inline static const Field& get_polynomial(const poly_array& polynomials, const size_t = 0)
    {
        if constexpr (use_shifted_evaluation) {
            return polynomials[id].second;
        }
        return polynomials[id].first;
    }

    static poly_array get_polynomial_evaluations(const polynomial_manifest& polynomial_manifest,
                                                 const Transcript& transcript)
    {
        poly_array result{};
        for (const auto& info : polynomial_manifest) {
            const std::string label(info.polynomial_label);
            if (!info.is_linearised || !Settings::use_linearisation) {
                result[info.index].first = transcript.get_field_element(label);
            } else {
                result[info.index].first = 0;
            }
            if (info.requires_shifted_evaluation) {
                result[info.index].second = transcript.get_field_element(label + "_omega");
            } else {
                result[info.index].second = 0;
            }
        }
        return result;
    }
};

template <typename Field, class Transcript, class Settings, size_t num_widget_relations>
class FFTGetter : public BaseGetter<Field, Transcript, Settings, num_widget_relations> {
  protected:
    typedef containers::poly_ptr_array<Field> poly_ptr_array;

  public:
    static poly_ptr_array get_fft_polynomials(proving_key* key)
    {
        poly_ptr_array result;
        result.block_mask = key->large_domain.size - 1;
        for (const auto& info : key->polynomial_manifest) {
            const std::string label = std::string(info.polynomial_label) + "_fft";
            Field* poly = nullptr;
            switch (info.source) {
            case PolynomialSource::WITNESS: {
                poly = &key->wire_ffts.at(label)[0];
                break;
            }
            case PolynomialSource::SELECTOR: {
                poly = &key->constraint_selector_ffts.at(label)[0];
                break;
            }
            case PolynomialSource::PERMUTATION: {
                poly = &key->permutation_selector_ffts.at(label)[0];
                break;
            }
            }
            result.coefficients[info.index] = poly;
        }
        return result;
    }

    template <bool use_shifted_evaluation, PolynomialIndex id>
    inline static const Field& get_polynomial(const poly_ptr_array& polynomials, const size_t index = 0)
    {
        // have range quotient R(X). Value of array are: R(g.w'^i) for i in [0, ...., 4n - 1], w' = 4n'th root of unity.
        // g = coset generator shifted range quotient = R(X.w), w = n'th root of unity. if we have at idx i: R(w'^i) =>
        // X = w'^i . Shifted equivalent = X.w = w'^i.w = w'^{i + 4} = value at index i + 4

        // final computation: T(X) = quotient identity I(X) / Z_H(X) <- vanishing poly

        if constexpr (use_shifted_evaluation) {
            return polynomials.coefficients[id][(index + 4) & polynomials.block_mask];
        }
        const Field& result = polynomials.coefficients[id][index];
        return result;
    }
};

template <typename Field, class Transcript, class Settings, size_t num_widget_relations>
class MonomialGetter : public BaseGetter<Field, Transcript, Settings, num_widget_relations> {
  protected:
    typedef containers::poly_ptr_array<Field> poly_ptr_array;

  public:
    static poly_ptr_array get_monomials(proving_key* key, program_witness* witness)
    {
        poly_ptr_array result;
        result.block_mask = key->small_domain.size - 1;
        for (const auto& info : key->polynomial_manifest) {
            const std::string label(info.polynomial_label);
            Field* poly = nullptr;
            switch (info.source) {
            case PolynomialSource::WITNESS: {
                poly = &witness->wires.at(label)[0];
                break;
            }
            case PolynomialSource::SELECTOR: {
                poly = &key->constraint_selectors.at(label)[0];
                break;
            }
            case PolynomialSource::PERMUTATION: {
                poly = &key->permutation_selectors.at(label)[0];
                break;
            }
            }
            result.coefficients[info.index] = poly;
        }
        return result;
    }

    template <bool use_shifted_evaluation, PolynomialIndex id>
    inline static Field& get_polynomial(const poly_ptr_array& polynomials, const size_t index = 0)
    {
        if constexpr (use_shifted_evaluation) {
            return polynomials.coefficients[id][(index + 1) & polynomials.block_mask];
        }
        return polynomials.coefficients[id][index];
    }
};
} // namespace getters

template <class Field> class TransitionWidgetBase {
  public:
    TransitionWidgetBase(proving_key* _key = nullptr, program_witness* _witness = nullptr)
        : key(_key)
        , witness(_witness){};
    TransitionWidgetBase(const TransitionWidgetBase& other)
        : key(other.key)
        , witness(other.witness){};
    TransitionWidgetBase(TransitionWidgetBase&& other)
        : key(other.key)
        , witness(other.witness){};
    TransitionWidgetBase& operator=(const TransitionWidgetBase& other)
    {
        key = other.key;
        witness = other.witness;
        return *this;
    };
    TransitionWidgetBase& operator=(TransitionWidgetBase&& other)
    {
        key = other.key;
        witness = other.witness;
        return *this;
    };
    virtual ~TransitionWidgetBase() {}

    virtual Field compute_quotient_contribution(const Field&, const transcript::StandardTranscript&) = 0;
    virtual Field compute_linear_contribution(const Field&, const transcript::StandardTranscript&, Field*) = 0;

  public:
    proving_key* key;
    program_witness* witness;
};

template <class Field, class Settings, template <typename, typename, typename> typename KernelBase>
class TransitionWidget : public TransitionWidgetBase<Field> {
  protected:
    static constexpr size_t num_independent_relations = KernelBase<int, int, int>::num_independent_relations;
    typedef containers::poly_ptr_array<Field> poly_ptr_array;
    typedef containers::poly_array<Field> poly_array;
    typedef containers::challenge_array<Field, num_independent_relations> challenge_array;
    typedef containers::coefficient_array<Field> coefficient_array;

  public:
    typedef getters::EvaluationGetter<Field, transcript::StandardTranscript, Settings, num_independent_relations>
        EvaluationGetter;
    typedef getters::FFTGetter<Field, transcript::StandardTranscript, Settings, num_independent_relations> FFTGetter;
    typedef getters::MonomialGetter<Field, transcript::StandardTranscript, Settings, num_independent_relations>
        MonomialGetter;
    typedef KernelBase<Field, FFTGetter, poly_ptr_array> FFTKernel;
    typedef KernelBase<Field, MonomialGetter, poly_ptr_array> MonomialKernel;
    typedef KernelBase<Field, EvaluationGetter, poly_array> EvaluationKernel;

    TransitionWidget(proving_key* _key = nullptr, program_witness* _witness = nullptr)
        : TransitionWidgetBase<Field>(_key, _witness){};
    TransitionWidget(const TransitionWidget& other)
        : TransitionWidgetBase<Field>(other){};
    TransitionWidget(TransitionWidget&& other)
        : TransitionWidgetBase<Field>(other){};
    TransitionWidget& operator=(const TransitionWidget& other)
    {
        TransitionWidgetBase<Field>::operator=(other);
        return *this;
    };
    TransitionWidget& operator=(TransitionWidget&& other)
    {
        TransitionWidgetBase<Field>::operator=(other);
        return *this;
    };

    Field compute_quotient_contribution(const Field& alpha_base,
                                        const transcript::StandardTranscript& transcript) override
    {
        auto* key = TransitionWidgetBase<Field>::key;

        poly_ptr_array polynomials = FFTGetter::get_fft_polynomials(key);
        challenge_array challenges =
            FFTGetter::get_challenges(transcript, alpha_base, FFTKernel::quotient_required_challenges);

        ITERATE_OVER_DOMAIN_START(key->large_domain);
        coefficient_array linear_terms;
        FFTKernel::compute_linear_terms(polynomials, challenges, linear_terms, i);
        Field sum_of_linear_terms = FFTKernel::sum_linear_terms(polynomials, challenges, linear_terms, i);

        // populate split quotient components
        Field& quotient_term = key->quotient_polynomial_parts[i >> key->small_domain.log2_size][i & (key->n - 1)];
        quotient_term += sum_of_linear_terms;
        FFTKernel::compute_non_linear_terms(polynomials, challenges, quotient_term, i);
        ITERATE_OVER_DOMAIN_END;

        return FFTGetter::update_alpha(challenges, FFTKernel::num_independent_relations);
    }

    Field compute_linear_contribution(const Field& alpha_base,
                                      const transcript::StandardTranscript& transcript,
                                      Field* linear_poly) override
    {
        challenge_array challenges = MonomialGetter::get_challenges(transcript,
                                                                    alpha_base,
                                                                    EvaluationKernel::quotient_required_challenges |
                                                                        MonomialKernel::quotient_required_challenges);

        if constexpr (!Settings::use_linearisation) {
            return MonomialGetter::update_alpha(challenges, FFTKernel::num_independent_relations);
        }
        auto* key = TransitionWidgetBase<Field>::key;
        auto* witness = TransitionWidgetBase<Field>::witness;

        poly_ptr_array polynomials = MonomialGetter::get_monomials(key, witness);
        poly_array polynomial_evaluations =
            EvaluationGetter::get_polynomial_evaluations(key->polynomial_manifest, transcript);

        coefficient_array linear_terms;
        EvaluationKernel::compute_linear_terms(polynomial_evaluations, challenges, linear_terms);

        ITERATE_OVER_DOMAIN_START(key->small_domain);
        linear_poly[i] += MonomialKernel::sum_linear_terms(polynomials, challenges, linear_terms, i);
        ITERATE_OVER_DOMAIN_END;

        if (Settings::use_linearisation) {
            EvaluationKernel::compute_non_linear_terms(polynomial_evaluations, challenges, linear_poly[0]);
        }

        return MonomialGetter::update_alpha(challenges, FFTKernel::num_independent_relations);
    }
};

template <class Field, class Transcript, class Settings, template <typename, typename, typename> typename KernelBase>
class GenericVerifierWidget {
  protected:
    static constexpr size_t num_independent_relations = KernelBase<int, int, int>::num_independent_relations;
    typedef containers::poly_ptr_array<Field> poly_ptr_array;
    typedef containers::poly_array<Field> poly_array;
    typedef containers::challenge_array<Field, num_independent_relations> challenge_array;
    typedef containers::coefficient_array<Field> coefficient_array;

  public:
    typedef getters::EvaluationGetter<Field, Transcript, Settings, num_independent_relations> EvaluationGetter;
    typedef KernelBase<Field, EvaluationGetter, poly_array> EvaluationKernel;

    static Field compute_quotient_evaluation_contribution(typename Transcript::Key* key,
                                                          const Field& alpha_base,
                                                          const Transcript& transcript,
                                                          Field& r_0)
    {
        poly_array polynomial_evaluations =
            EvaluationGetter::get_polynomial_evaluations(key->polynomial_manifest, transcript);
        challenge_array challenges =
            EvaluationGetter::get_challenges(transcript, alpha_base, EvaluationKernel::quotient_required_challenges);

        if constexpr (!Settings::use_linearisation) {
            coefficient_array linear_terms;
            EvaluationKernel::compute_linear_terms(polynomial_evaluations, challenges, linear_terms);
            r_0 += EvaluationKernel::sum_linear_terms(polynomial_evaluations, challenges, linear_terms);
            EvaluationKernel::compute_non_linear_terms(polynomial_evaluations, challenges, r_0);
        } else {
            EvaluationKernel::compute_non_linear_terms(polynomial_evaluations, challenges, r_0);
        }
        return EvaluationGetter::update_alpha(challenges, num_independent_relations);
    }

    static Field append_scalar_multiplication_inputs(typename Transcript::Key* key,
                                                     const Field& alpha_base,
                                                     const Transcript& transcript,
                                                     std::map<std::string, Field>& scalars)
    {
        challenge_array challenges = EvaluationGetter::get_challenges(transcript,
                                                                      alpha_base,
                                                                      EvaluationKernel::quotient_required_challenges |
                                                                          EvaluationKernel::update_required_challenges);
        if (!Settings::use_linearisation) {
            return EvaluationGetter::update_alpha(challenges, num_independent_relations);
        }

        poly_array polynomial_evaluations =
            EvaluationGetter::get_polynomial_evaluations(key->polynomial_manifest, transcript);

        coefficient_array linear_terms;
        EvaluationKernel::compute_linear_terms(polynomial_evaluations, challenges, linear_terms);

        EvaluationKernel::update_kate_opening_scalars(linear_terms, scalars, challenges);

        return EvaluationGetter::update_alpha(challenges, num_independent_relations);
    }
};
} // namespace widget
} // namespace waffle