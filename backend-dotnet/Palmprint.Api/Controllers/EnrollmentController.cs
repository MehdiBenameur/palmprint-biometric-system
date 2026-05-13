using Microsoft.AspNetCore.Mvc;
using Palmprint.Api.Models;
using Palmprint.Application.DTOs;
using Palmprint.Application.Interfaces;

namespace Palmprint.Api.Controllers;

[ApiController]
[Route("api/enrollment")]
public class EnrollmentController : ControllerBase
{
    private readonly IEnrollmentService _enrollmentService;

    public EnrollmentController(IEnrollmentService enrollmentService)
    {
        _enrollmentService = enrollmentService;
    }

    [HttpPost("enroll")]
    [Consumes("multipart/form-data")]
    public async Task<IActionResult> Enroll([FromForm] EnrollFormRequest form)
    {
        if (form.TenantId == Guid.Empty)
            return BadRequest("TenantId is required.");

        if (string.IsNullOrWhiteSpace(form.FullName))
            return BadRequest("FullName is required.");

        if (string.IsNullOrWhiteSpace(form.ExternalId))
            return BadRequest("ExternalId is required.");

        if (form.Image == null || form.Image.Length == 0)
            return BadRequest("Image is required.");

        var allowedExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif" };
        var extension = Path.GetExtension(form.Image.FileName).ToLowerInvariant();

        if (!allowedExtensions.Contains(extension))
            return BadRequest("Invalid image format. Allowed formats: jpg, jpeg, png, bmp, tiff, tif.");

        const long maxFileSize = 5 * 1024 * 1024;

        if (form.Image.Length > maxFileSize)
            return BadRequest("Image size exceeds the maximum allowed size of 5 MB.");

        try
        {
            using var memoryStream = new MemoryStream();
            await form.Image.CopyToAsync(memoryStream);

            var request = new EnrollRequest
            {
                TenantId = form.TenantId,
                FullName = form.FullName,
                ExternalId = form.ExternalId,
                ImageBytes = memoryStream.ToArray()
            };

            var response = await _enrollmentService.EnrollAsync(request);

            return Ok(response);
        }
        catch (ArgumentException ex)
        {
            return BadRequest(ex.Message);
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(ex.Message);
        }
        catch
        {
            return StatusCode(500, "An unexpected error occurred during enrollment.");
        }
    }
}