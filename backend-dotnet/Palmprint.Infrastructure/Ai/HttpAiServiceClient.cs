using System.Net.Http.Headers;
using Newtonsoft.Json;
using Palmprint.Application.Interfaces;

namespace Palmprint.Infrastructure.Ai;

public class HttpAiServiceClient : IAiServiceClient
{
    private readonly HttpClient _httpClient;

    public HttpAiServiceClient(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<(float[] CnnEmbedding, float[] TripletEmbedding, double QualityScore, string ModelVersion)>
        GenerateEmbeddingAsync(byte[] imageBytes)
    {
        using var content = new MultipartFormDataContent();

        var imageContent = new ByteArrayContent(imageBytes);

        imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/bmp");

        content.Add(imageContent, "image", "palm.bmp");

        var response = await _httpClient.PostAsync(
            "http://127.0.0.1:8000/generate-embedding",
            content
        );

        var json = await response.Content.ReadAsStringAsync();

        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException(
                $"AI Service Error: {json}"
            );
        }

        var result = JsonConvert.DeserializeObject<GenerateEmbeddingResponse>(json);

        if (result == null)
            throw new InvalidOperationException("Invalid AI response.");

        return (
            result.CnnEmbedding,
            result.TripletEmbedding,
            result.QualityScore,
            result.ModelVersion
        );
    }
}