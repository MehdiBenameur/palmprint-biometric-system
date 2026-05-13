namespace Palmprint.Application.Interfaces;

public interface IAiServiceClient
{
    Task<(float[] CnnEmbedding, float[] TripletEmbedding, double QualityScore, string ModelVersion)> GenerateEmbeddingAsync(byte[] imageBytes);
}