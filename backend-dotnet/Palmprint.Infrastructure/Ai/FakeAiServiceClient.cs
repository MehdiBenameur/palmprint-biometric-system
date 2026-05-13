using Palmprint.Application.Interfaces;

namespace Palmprint.Infrastructure.Ai;

public class FakeAiServiceClient : IAiServiceClient
{
    public Task<(float[] Embedding, double QualityScore, string ModelVersion)> GenerateEmbeddingAsync(byte[] imageBytes)
    {
        var random = new Random();

        var embedding = new float[1792];

        for (int i = 0; i < embedding.Length; i++)
        {
            embedding[i] = (float)random.NextDouble();
        }

        return Task.FromResult((
            embedding,
            0.90,
            "fake-cnn-triplet-fusion-v1"
        ));
    }
}