// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.Linq;
using SmartComponents.Inference;

namespace SmartComponents.LocalEmbeddings;

public partial class LocalEmbedder
{
    public static float Similarity<TEmbedding>(TEmbedding a, TEmbedding b) where TEmbedding : IEmbedding<TEmbedding>
        => a.Similarity(b);

    /// <summary>
    /// Finds the closest matching items to the specified query based on similarity.
    /// </summary>
    /// <typeparam name="TItem">The type of the items being searched.</typeparam>
    /// <typeparam name="TEmbedding">The type of the embeddings used for similarity comparison. Must implement <see cref="IEmbedding{TEmbedding}"/>.</typeparam>
    /// <param name="query">The similarity query containing the search text, maximum results, and minimum similarity threshold.</param>
    /// <param name="candidates">A collection of candidate items and their corresponding embeddings to compare against the query.</param>
    /// <returns>An array of items that are the closest matches to the query, ordered by similarity. The array will contain at
    /// most <paramref name="query.MaxResults"/> items and will exclude items with a similarity below <paramref
    /// name="query.MinSimilarity"/>.</returns>
    public TItem[] FindClosest<TItem, TEmbedding>(
        SimilarityQuery query,
        IEnumerable<(TItem Item, TEmbedding Embedding)> candidates) where TEmbedding : IEmbedding<TEmbedding>
        => [.. FindClosestCore(Embed<TEmbedding>(query.SearchText), candidates, query.MaxResults, query.MinSimilarity).Select(x => x.Item)];

    /// <summary>
    /// Finds the closest items to the specified query, along with their similarity scores.
    /// </summary>
    /// <remarks>This method computes the similarity between the query's embedding and the embeddings of the
    /// provided candidates. Only items with a similarity score greater than or equal to the specified minimum
    /// similarity are included in the results.</remarks>
    /// <typeparam name="TItem">The type of the items being compared.</typeparam>
    /// <typeparam name="TEmbedding">The type of the embeddings used for similarity comparison. Must implement <see cref="IEmbedding{TEmbedding}"/>.</typeparam>
    /// <param name="query">The similarity query containing the search text, maximum results, and minimum similarity threshold.</param>
    /// <param name="candidates">A collection of items and their corresponding embeddings to compare against the query.</param>
    /// <returns>An array of <see cref="SimilarityScore{TItem}"/> objects, each representing an item and its similarity score.
    /// The array is sorted in descending order of similarity.</returns>
    public SimilarityScore<TItem>[] FindClosestWithScore<TItem, TEmbedding>(
        SimilarityQuery query,
        IEnumerable<(TItem Item, TEmbedding Embedding)> candidates) where TEmbedding : IEmbedding<TEmbedding>
        => [.. FindClosestCore(Embed<TEmbedding>(query.SearchText), candidates, query.MaxResults, query.MinSimilarity)];

    /// <summary>
    /// Finds the closest <paramref name="maxResults"/> candidates to <paramref name="target"/>.
    /// </summary>
    /// <typeparam name="TItem">The type of the items being searched</typeparam>
    /// <typeparam name="TEmbedding">The type of the embeddings</typeparam>
    /// <param name="target">An embedding representing the value to be searched for.</param>
    /// <param name="candidates">The set of possible results.</param>
    /// <param name="maxResults">Specifies an upper limit on the number of results. If no limit is required, pass <see cref="int.MaxValue"/>.</param>
    /// <param name="minSimilarity">Specifies a lower limit on the similarity ranking for matching results.</param>
    /// <returns>An ordered array of <typeparamref name="TItem"/> values, starting from the most similar.</returns>
    public static TItem[] FindClosest<TItem, TEmbedding>(
        TEmbedding target,
        IEnumerable<(TItem Item, TEmbedding Embedding)> candidates,
        int maxResults,
        float? minSimilarity = null) where TEmbedding : IEmbedding<TEmbedding>
        => [.. FindClosestCore(target, candidates, maxResults, minSimilarity).Select(x => x.Item)];

    /// <summary>
    /// Finds the closest <paramref name="maxResults"/> candidates to <paramref name="target"/>,
    /// returning both the similarity score and the corresponding value.
    /// </summary>
    /// <typeparam name="TItem">The type of the items being searched</typeparam>
    /// <typeparam name="TEmbedding">The type of the embeddings</typeparam>
    /// <param name="target">An embedding representing the value to be searched for.</param>
    /// <param name="candidates">The set of possible results.</param>
    /// <param name="maxResults">Specifies an upper limit on the number of results. If no limit is required, pass <see cref="int.MaxValue"/>.</param>
    /// <param name="minSimilarity">Specifies a lower limit on the similarity ranking for matching results.</param>
    /// <returns>An ordered array of <see cref="SimilarityScore{T}"/> values that specify the similarity along with the corresponding <typeparamref name="TItem"/> value.</returns>
    public static SimilarityScore<TItem>[] FindClosestWithScore<TItem, TEmbedding>(
        TEmbedding target,
        IEnumerable<(TItem Item, TEmbedding Embedding)> candidates,
        int maxResults,
        float? minSimilarity = null) where TEmbedding : IEmbedding<TEmbedding>
        => [.. FindClosestCore(target, candidates, maxResults, minSimilarity)];

    private static SortedSet<SimilarityScore<TItem>> FindClosestCore<TItem, TEmbedding>(
        TEmbedding target,
        IEnumerable<(TItem Item, TEmbedding Embedding)> candidates,
        int maxResults,
        float? minSimilarity = null) where TEmbedding : IEmbedding<TEmbedding>
    {
        if (maxResults <= 0)
        {
            throw new ArgumentException($"{maxResults} must be greater than 0.");
        }

        var sortedTopK = new SortedSet<SimilarityScore<TItem>>(SimilarityScore<TItem>._comparer);
        var candidatesEnumerator = candidates.GetEnumerator();
        var index = 0L;
        minSimilarity ??= float.MinValue;

        // Populate the results with the first K candidates
        while (sortedTopK.Count < maxResults && candidatesEnumerator.MoveNext())
        {
            var (Item, Embedding) = candidatesEnumerator.Current;
            var similarity = target.Similarity(Embedding);
            if (similarity >= minSimilarity)
            {
                sortedTopK.Add(new SimilarityScore<TItem>(similarity, Item, index++));
            }
        }

        // Add remaining candidates only if they are better than the worst so far
        while (candidatesEnumerator.MoveNext())
        {
            var (Item, Embedding) = candidatesEnumerator.Current;
            var similarity = target.Similarity(Embedding);

            // By this point we know there's a nonzero number of elements in the set
            // so we can just compare against the worst so far ("Max"), and can ignore
            // the minSimilarity threshold
            if (similarity > sortedTopK.Max.Similarity)
            {
                sortedTopK.Remove(sortedTopK.Max);
                sortedTopK.Add(new SimilarityScore<TItem>(similarity, Item, index++));
            }
        }

        return sortedTopK;
    }
}
